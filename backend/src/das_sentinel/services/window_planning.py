"""Prepare continuous window plans outside the API's database write lock."""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
import math
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

from sqlalchemy import select

from das_sentinel.persistence.models import SourceFile
from das_sentinel.persistence.repositories import SourceFileRepository
from das_sentinel.services.discovery import (
    DiscoveredFile,
    DiscoveryConfigurationError,
    DiscoveryPathError,
    _open_anchored_source,
    load_discovery_site,
)
from das_sentinel.whole_window import (
    ALGORITHM_VERSION,
    READ_CONTEXT_SECONDS,
    WINDOW_SECONDS,
    inspect_sources,
    plan_windows,
)

logger = logging.getLogger(__name__)


def deduplicate_sources(metadata, cache):
    """Collapse byte-identical copies only; keep conflicting recordings visible."""
    groups = {}
    for source in metadata:
        key = json.dumps({k: source[k] for k in (
            'start_utc', 'end_utc', 'sample_rate_hz', 'distance_m', 'unit', 'size_bytes',
        )}, sort_keys=True)
        groups.setdefault(key, []).append(source)
    selected = []
    for sources in groups.values():
        if len(sources) == 1:
            selected.extend(sources)
            continue
        seen = set()
        for source in sorted(sources, key=lambda s: s['path']):
            path = Path(source['path'])
            expected = DiscoveredFile(path, source['fingerprint'], source['size_bytes'], source['mtime_ns'])
            if DiscoveredFile.from_path(path) != expected:
                raise DiscoveryPathError('source changed before duplicate check')
            cache = Path(cache)
            cache.mkdir(parents=True, exist_ok=True)
            cached = cache / f"{source['fingerprint']}.json"
            if cached.exists():
                signature = json.loads(cached.read_text(encoding='utf8'))['sha256']
            else:
                with path.open('rb') as stream:
                    signature = hashlib.file_digest(stream, 'sha256').hexdigest()
                if DiscoveredFile.from_path(path) != expected:
                    raise DiscoveryPathError('source changed during duplicate check')
                temporary = cache / f'{uuid4()}.tmp'
                temporary.write_text(json.dumps({'sha256': signature}), encoding='utf8')
                os.replace(temporary, cached)
            if signature not in seen:
                selected.append(source)
                seen.add(signature)
    return selected


def _neighbor_snapshots(files, root):
    """Locate bounded adjacent minute files; their headers remain authoritative."""
    seen = {str(source.path) for source in files}
    found = []
    for source in files:
        path = Path(source.path)
        match = re.fullmatch(r"(.*?)(\d{12})", path.stem)
        if match is None:
            continue
        try:
            # A filename calendar hint, never the authoritative source clock.
            stamp = datetime.strptime(match[2], "%Y%m%d%H%M")  # noqa: DTZ007
        except ValueError:
            continue
        for offset in (-2, -1, 1, 2):
            neighbor = stamp + timedelta(minutes=offset)
            name = f"{match[1]}{neighbor:%Y%m%d%H%M}{path.suffix}"
            parents = [path.parent]
            if re.fullmatch(r"\d{8}|\d{10}", path.parent.name):
                pattern = "%Y%m%d" if len(path.parent.name) == 8 else "%Y%m%d%H"
                parents.append(path.parent.parent / neighbor.strftime(pattern))
            for parent in dict.fromkeys(parents):
                candidate = parent / name
                if not candidate.is_relative_to(root):
                    continue
                if str(candidate) in seen:
                    break
                if not candidate.is_file():
                    continue
                with _open_anchored_source(root, candidate) as anchored:
                    found.append(anchored.snapshot().discovered())
                seen.add(str(candidate))
                break
    return found


def _cached_metadata_matches(value, source):
    if not isinstance(value, dict) or any(value.get(key) != expected for key, expected in (
        ("path", str(source.path)), ("fingerprint", source.fingerprint),
        ("size_bytes", source.size_bytes), ("mtime_ns", source.mtime_ns),
    )):
        return False
    try:
        start, end = (datetime.fromisoformat(value[key]) for key in ("start_utc", "end_utc"))
        rate, distances = value["sample_rate_hz"], value["distance_m"]
        return (
            start.utcoffset() is not None and end.utcoffset() is not None and end > start
            and type(rate) in (int, float) and math.isfinite(rate) and rate >= 100
            and isinstance(distances, list) and bool(distances)
            and all(type(distance) in (int, float) and math.isfinite(distance) for distance in distances)
            and all(right > left for left, right in itertools.pairwise(distances))
            and isinstance(value["unit"], str) and bool(value["unit"].strip())
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def _inspect(source, root, cache):
    path = Path(source.path)
    with _open_anchored_source(root, path, required_suffix=path.suffix) as anchored:
        before = anchored.snapshot().discovered()
        if (before.fingerprint, before.size_bytes, before.mtime_ns) != (
            source.fingerprint,
            source.size_bytes,
            source.mtime_ns,
        ):
            raise DiscoveryPathError("source changed since discovery")
        cached = cache / f"{source.fingerprint}.json"
        if cached.is_file():
            try:
                value = json.loads(cached.read_text(encoding="utf-8"))
            except ValueError:
                value = None
            if _cached_metadata_matches(value, source):
                if anchored.snapshot().discovered() != before:
                    raise DiscoveryPathError("source changed while reading metadata cache")
                return value
        value = inspect_sources([path])[0]
        after = anchored.snapshot().discovered()
        if after != before:
            raise DiscoveryPathError("source changed during window planning")
        cache.mkdir(parents=True, exist_ok=True)
        temporary = cache / f"{uuid4()}.tmp"
        temporary.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")
        os.replace(temporary, cached)
        return value


def prepare_window_files(sessions, site, files, artifact_root):
    """Return requested windows, finding context without expanding the request."""
    if not files:
        return []
    with sessions() as session:
        current = load_discovery_site(session, site.site_id)
        if (current.profile_id, current.profile_version) != (
            site.profile_id,
            site.profile_version,
        ):
            raise DiscoveryConfigurationError(
                "current processing profile changed after discovery"
            )
        registered = session.scalars(
            select(SourceFile)
            .where(
                SourceFile.site_id == site.site_id,
            )
            .order_by(SourceFile.discovered_at.desc())
        ).all()
    selected = {str(source.path): source for source in files}
    if any(not isinstance(source, DiscoveredFile) for source in selected.values()):
        raise DiscoveryPathError("window planning requires stable source snapshots")
    # A source can have historical fingerprints. Choose only its most recent
    # registration; an explicitly selected stable snapshot takes precedence.
    sources = dict(selected)
    for source in _neighbor_snapshots(files, current.data_root):
        sources.setdefault(str(source.path), source)
    for row in registered:
        sources.setdefault(
            row.path,
            DiscoveredFile(
                Path(row.path),
                row.fingerprint,
                row.size_bytes,
                row.mtime_ns,
            ),
        )
    cache = Path(artifact_root).parent / f"window-source-metadata-{ALGORITHM_VERSION}"
    metadata, requested = [], []
    for path, source in sources.items():
        try:
            item = _inspect(source, current.data_root, cache)
        except (OSError, ValueError):
            if path in selected:
                raise
            logger.warning(
                "Registered context source unavailable during window planning"
            )
            continue
        metadata.append(item)
        if path in selected:
            requested.append(
                (
                    datetime.fromisoformat(item["start_utc"]),
                    datetime.fromisoformat(item["end_utc"]),
                )
            )
    # Include alignment and read context, but never hash unrelated historical copies.
    margin = timedelta(seconds=READ_CONTEXT_SECONDS + WINDOW_SECONDS)
    metadata = [item for item in metadata if any(
        datetime.fromisoformat(item['start_utc']) < last + margin
        and datetime.fromisoformat(item['end_utc']) > first - margin
        for first, last in requested
    )]
    plans = []
    for plan in plan_windows(deduplicate_sources(metadata, cache / 'content-hashes')):
        start, end = (
            datetime.fromisoformat(plan[key]) for key in ("start_utc", "end_utc")
        )
        if any(start < last and end > first for first, last in requested):
            plans.append(plan)
    return plans


def validate_window_sources(site, files, plans):
    """Recheck selected and context snapshots immediately before enqueueing."""
    snapshots = {str(source.path): source for source in files}
    for plan in plans:
        for source in plan["sources"]:
            snapshots.setdefault(
                source["path"],
                DiscoveredFile(
                    Path(source["path"]),
                    source["fingerprint"],
                    source["size_bytes"],
                    source["mtime_ns"],
                ),
            )
    for source in snapshots.values():
        with _open_anchored_source(site.data_root, source.path) as anchored:
            if anchored.snapshot().discovered() != source:
                raise DiscoveryPathError("source changed before window enqueue")


def persist_window_files(session, site, files, plans, *, sources_validated=False):
    from das_sentinel.services.event_windows import enqueue_windows

    current = load_discovery_site(session, site.site_id)
    if (current.profile_id, current.profile_version) != (
        site.profile_id,
        site.profile_version,
    ):
        raise DiscoveryConfigurationError(
            "current processing profile changed during planning"
        )
    if not sources_validated:
        validate_window_sources(current, files, plans)
    repository = SourceFileRepository(session)
    for source in files:
        if (
            repository.find(site_id=site.site_id, fingerprint=source.fingerprint)
            is None
        ):
            repository.add(
                site_id=site.site_id,
                path=str(source.path),
                fingerprint=source.fingerprint,
                size_bytes=source.size_bytes,
                mtime_ns=source.mtime_ns,
            )
    return enqueue_windows(session, site.site_id, site.profile_id, plans)


def enqueue_window_files(sessions, site, files, artifact_root):
    plans = prepare_window_files(sessions, site, files, artifact_root)
    validate_window_sources(site, files, plans)
    with sessions.begin() as session:
        return persist_window_files(session, site, files, plans, sources_validated=True)

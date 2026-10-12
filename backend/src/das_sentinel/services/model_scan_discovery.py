"""Read-only, bounded NAS discovery and planning outside database transactions."""

import os
import time
from datetime import timedelta
from pathlib import Path

from das_sentinel.model_scan_contract import excluded_two_metre_source, utc
from das_sentinel.services.discovery import (
    DiscoveredFile,
    _is_reparse,
    _open_anchored_source,
    _validate_under_root,
    _validated_root,
)
from das_sentinel.services.window_planning import _inspect, deduplicate_sources
from das_sentinel.whole_window import inspect_sources, plan_windows


def discover_directory(root: Path, directory: Path, *, stable_seconds=10.0, cache_root=None):
    root = _validated_root(Path(root))
    directory = _validate_under_root(root, Path(directory), directory=True)
    # Paths/stat records only; no per-file waveform or full metadata inventory in RAM.
    files, children = [], []
    with os.scandir(directory) as entries:
        for entry in entries:
            path = Path(entry.path)
            if entry.name.startswith(("@", ".")) or _is_reparse(path):
                continue
            if entry.is_dir(follow_symlinks=False):
                children.append(str(path))
            elif entry.is_file(follow_symlinks=False) and path.suffix.lower() == ".h5":
                files.append(DiscoveredFile.from_path(path))
    files.sort(key=lambda item: str(item.path).casefold())
    children.sort(key=str.casefold)
    if files and stable_seconds:
        time.sleep(stable_seconds)
    page = {"sources": [], "directories": [], "errors": []}
    for child in children:
        page["directories"].append(child)
        if sum(map(len, page.values())) >= 256:
            yield page
            page = {"sources": [], "directories": [], "errors": []}
    for before in files:
        try:
            with _open_anchored_source(root, before.path) as anchored:
                if anchored.snapshot().discovered() != before:
                    raise ValueError("source changed during stability interval")
                source = (_inspect(before, root, Path(cache_root)) if cache_root is not None
                          else inspect_sources([before.path])[0])
                if anchored.snapshot().discovered() != before:
                    raise ValueError("source changed during metadata inspection")
            page["sources"].append(source)
        except (OSError, ValueError) as error:
            page["errors"].append({"path": str(before.path), "error": str(error)[:1000]})
        if sum(map(len, page.values())) >= 256:
            yield page
            page = {"sources": [], "directories": [], "errors": []}
    if any(page.values()):
        yield page


def discover_context(root, paths, *, cache_root=None):
    page = {"sources": [], "directories": [], "errors": []}
    for path in paths:
        try:
            with _open_anchored_source(Path(root), Path(path)) as anchored:
                before = anchored.snapshot().discovered()
                source = (_inspect(before, Path(root), Path(cache_root)) if cache_root is not None
                          else inspect_sources([Path(path)])[0])
                if anchored.snapshot().discovered() != before:
                    raise ValueError("registered source changed during inspection")
                page["sources"].append(source)
        except (OSError, ValueError) as error:
            page["errors"].append({"path": path, "error": str(error)[:1000]})
    yield page


def plan_source_windows(sources, *, block_seconds=300, cache_root, start_utc=None, end_utc=None, requested_sources=None):
    if block_seconds < 30 or block_seconds > 300 or block_seconds % 30:
        raise ValueError("blocks must be a multiple of 30 seconds, at most 300")
    required = {s["path"] for s in (requested_sources if requested_sources is not None else sources)}
    excluded = [s for s in sources if excluded_two_metre_source(s)]
    sources = [s for s in sources if not excluded_two_metre_source(s)]
    available, invalid = [], []
    for source in sources:
        try:
            expected = DiscoveredFile(Path(source["path"]), source["fingerprint"], source["size_bytes"], source["mtime_ns"])
            if DiscoveredFile.from_path(expected.path) != expected:
                raise ValueError("source changed before window planning")
            available.append(source)
        except (OSError, ValueError):
            if source["path"] in required:
                invalid.append(source)
    try:
        sources = (deduplicate_sources(available, Path(cache_root) / "content-hashes") if available else []) + invalid
    except (OSError, ValueError):
        # A duplicate may change while it is hashed. Preserve its covered window
        # identities as unjudgeable, so later time blocks can still be processed.
        sources, invalid = available + invalid, available + invalid
    # Keep excluded time identities for auditable receipts, but never use their
    # channels as context for a normal recording covering the same time.
    by_start = {p["start_utc"]: p for p in plan_windows(excluded, 30)}
    by_start.update({p["start_utc"]: p for p in plan_windows(sources, 30)})
    plans = [by_start[start] for start in sorted(by_start)]
    invalid_paths = {s["path"] for s in invalid}
    for plan in plans:
        if any(s["path"] in invalid_paths for s in plan["sources"]):
            plan["coverage"] = "incomplete"
            plan["coverage_error"] = "source_changed_or_unavailable"
    if start_utc is not None:
        plans = [p for p in plans if utc(p["start_utc"]) >= utc(start_utc)]
    if end_utc is not None:
        plans = [p for p in plans if utc(p["end_utc"]) <= utc(end_utc)]
    if requested_sources is not None:
        plans = [p for p in plans if any(utc(p["start_utc"]) < utc(s["end_utc"]) and
            utc(p["end_utc"]) > utc(s["start_utc"]) for s in requested_sources)]
    blocks, current = [], []
    for plan in plans:
        if current and (utc(plan["end_utc"]) - utc(current[0]["start_utc"]) > timedelta(seconds=block_seconds)):
            blocks.append({"windows": current})
            current = []
        current.append(plan)
    if current:
        blocks.append({"windows": current})
    return blocks

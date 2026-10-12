from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool
from pydantic import ValidationError
from sqlalchemy.exc import IntegrityError

from das_sentinel.api.auth import AuthenticatedAdmin, CsrfAdmin
from das_sentinel.api.schemas import (
    ProfileResponse,
    SiteCreate,
    SitePatch,
    SiteResponse,
)
from das_sentinel.config import ProcessingProfile
from das_sentinel.persistence.models import ProcessingProfile as ProfileRow
from das_sentinel.persistence.models import Site
from das_sentinel.persistence.repositories import Repositories

router = APIRouter(prefix="/sites", tags=["sites"])


@router.post("", response_model=SiteResponse, status_code=status.HTTP_201_CREATED)
async def create_site(
    payload: SiteCreate,
    request: Request,
    admin: CsrfAdmin,
) -> SiteResponse:
    del admin
    coordinator = request.app.state.write_coordinator
    sessions = request.app.state.session_factory
    try:
        async with coordinator.lock:
            response = await run_in_threadpool(
                _create_site,
                sessions,
                request.app.state.settings.data_root,
                payload,
            )
    except IntegrityError as error:
        raise _conflict() from error
    return response


@router.get("", response_model=list[SiteResponse])
def list_sites(
    request: Request,
    admin: AuthenticatedAdmin,
) -> list[SiteResponse]:
    del admin
    sessions = request.app.state.session_factory
    with sessions() as session:
        repositories = Repositories(session)
        return [
            _site_response(site, _current_profile(repositories, site.id))
            for site in repositories.sites.list()
        ]


@router.get("/{site_id}", response_model=SiteResponse)
def get_site(
    site_id: str,
    request: Request,
    admin: AuthenticatedAdmin,
) -> SiteResponse:
    del admin
    sessions = request.app.state.session_factory
    with sessions() as session:
        repositories = Repositories(session)
        site = repositories.sites.get(site_id)
        if site is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Site not found",
            )
        return _site_response(site, _current_profile(repositories, site.id))


@router.patch("/{site_id}", response_model=SiteResponse)
async def update_site(
    site_id: str,
    payload: SitePatch,
    request: Request,
    admin: CsrfAdmin,
) -> SiteResponse:
    del admin
    coordinator = request.app.state.write_coordinator
    sessions = request.app.state.session_factory
    try:
        async with coordinator.lock:
            response = await run_in_threadpool(
                _update_site,
                sessions,
                request.app.state.settings.data_root,
                site_id,
                payload,
            )
    except IntegrityError as error:
        raise _conflict() from error
    return response


def _create_site(sessions, data_root: Path, payload: SiteCreate) -> SiteResponse:
    data_path = _resolve_data_path(data_root, payload.data_path)
    with sessions.begin() as session:
        repositories = Repositories(session)
        site = repositories.sites.add(
            name=payload.name,
            data_path=str(data_path),
            timezone=payload.timezone,
            cable_type=payload.cable_type,
            enabled=payload.enabled,
        )
        profile = repositories.profiles.add(
            site_id=site.id,
            version=1,
            payload=payload.profile.model_dump(mode="json"),
            is_current=True,
        )
        return _site_response(site, profile)


def _update_site(
    sessions,
    data_root: Path,
    site_id: str,
    payload: SitePatch,
) -> SiteResponse:
    resolved_path = (
        _resolve_data_path(data_root, payload.data_path)
        if payload.data_path is not None
        else None
    )
    with sessions.begin() as session:
        repositories = Repositories(session)
        site = repositories.sites.get(site_id)
        if site is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Site not found",
            )
        current = _current_profile(repositories, site.id)
        profile_updates = (
            payload.profile.model_dump(exclude_none=True)
            if payload.profile is not None
            else {}
        )
        if payload.profile is not None:
            for name in ("scan_distance_start_m", "scan_distance_stop_m"):
                if name in payload.profile.model_fields_set:
                    profile_updates[name] = getattr(payload.profile, name)
        try:
            next_profile = ProcessingProfile.model_validate(
                {**json.loads(current.payload_json), **profile_updates}
            )
        except ValidationError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="Invalid processing profile",
            ) from error

        if payload.name is not None:
            site.name = payload.name
        if resolved_path is not None:
            site.data_path = str(resolved_path)
        if payload.timezone is not None:
            site.timezone = payload.timezone
        if payload.cable_type is not None:
            site.cable_type = payload.cable_type
        if payload.enabled is not None:
            site.enabled = payload.enabled
        session.flush()

        new_profile = repositories.profiles.add(
            site_id=site.id,
            version=current.version + 1,
            payload=next_profile.model_dump(mode="json"),
            is_current=False,
        )
        repositories.sites.set_current_profile(site.id, new_profile.id)
        return _site_response(site, new_profile)


def _resolve_data_path(data_root_value: Path, submitted: str) -> Path:
    try:
        data_root = data_root_value.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Configured data root must be an existing directory",
        ) from error
    if not data_root.is_dir():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Configured data root must be an existing directory",
        )

    candidate = Path(submitted)
    if not candidate.is_absolute():
        candidate = data_root / candidate
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Site data path must be an existing directory",
        ) from error
    if not resolved.is_dir() or not resolved.is_relative_to(data_root):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Site data path must be a directory under the data root",
        )
    return resolved


def _current_profile(
    repositories: Repositories,
    site_id: str,
) -> ProfileRow:
    current = [
        profile
        for profile in repositories.profiles.list_for_site(site_id)
        if profile.is_current
    ]
    if len(current) != 1:
        raise RuntimeError("site must have exactly one current profile")
    return current[0]


def _site_response(site: Site, profile: ProfileRow) -> SiteResponse:
    return SiteResponse(
        id=site.id,
        name=site.name,
        data_path=site.data_path,
        timezone=site.timezone,
        cable_type=site.cable_type,
        enabled=site.enabled,
        created_at=site.created_at,
        current_profile=ProfileResponse(
            id=profile.id,
            version=profile.version,
            payload=json.loads(profile.payload_json),
            is_current=profile.is_current,
            created_at=profile.created_at,
        ),
    )


def _conflict() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail="Site name or profile version conflicts with existing data",
    )

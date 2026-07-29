from __future__ import annotations

from secrets import token_urlsafe

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError
from fastapi import APIRouter, HTTPException, Request, Response, status

from das_sentinel.api.auth import (
    SESSION_ADMIN_ID,
    SESSION_CSRF_TOKEN,
    AuthenticatedAdmin,
    CsrfAdmin,
)
from das_sentinel.api.schemas import LoginRequest, SessionResponse
from das_sentinel.persistence.repositories import Repositories

router = APIRouter(tags=["session"])
_PASSWORD_HASHER = PasswordHasher()
_DUMMY_HASH = _PASSWORD_HASHER.hash("not-a-real-admin-password")


@router.post("/session", status_code=status.HTTP_204_NO_CONTENT)
async def login(request: Request, credentials: LoginRequest) -> Response:
    request.session.clear()
    sessions = request.app.state.session_factory
    with sessions() as session:
        admin = Repositories(session).admins.get()
        password_hash = admin.password_hash if admin is not None else _DUMMY_HASH
        try:
            password_ok = _PASSWORD_HASHER.verify(
                password_hash,
                credentials.password.get_secret_value(),
            )
        except (VerificationError, InvalidHashError):
            password_ok = False

    if admin is None or admin.disabled or not password_ok:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized",
        )

    csrf_token = token_urlsafe(32)
    request.session[SESSION_ADMIN_ID] = admin.id
    request.session[SESSION_CSRF_TOKEN] = csrf_token
    return Response(
        status_code=status.HTTP_204_NO_CONTENT,
        headers={"X-CSRF-Token": csrf_token},
    )


@router.get("/session", response_model=SessionResponse)
async def get_session(
    request: Request,
    admin: AuthenticatedAdmin,
) -> SessionResponse:
    csrf_token = request.session.get(SESSION_CSRF_TOKEN)
    if not isinstance(csrf_token, str) or not csrf_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized",
        )
    return SessionResponse(admin_id=admin.id, csrf_token=csrf_token)


@router.delete("/session", status_code=status.HTTP_204_NO_CONTENT)
async def logout(
    request: Request,
    admin: CsrfAdmin,
) -> Response:
    del admin
    request.session.clear()
    return Response(status_code=status.HTTP_204_NO_CONTENT)

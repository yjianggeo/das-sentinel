from __future__ import annotations

import hmac
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status

from das_sentinel.persistence.models import Admin
from das_sentinel.persistence.repositories import Repositories

SESSION_ADMIN_ID = "admin_id"
SESSION_CSRF_TOKEN = "csrf_token"


def require_admin(request: Request) -> Admin:
    admin_id = request.session.get(SESSION_ADMIN_ID)
    if not isinstance(admin_id, int):
        raise _unauthorized()

    sessions = request.app.state.session_factory
    with sessions() as session:
        admin = Repositories(session).admins.get(admin_id)
        if admin is None or admin.disabled:
            raise _unauthorized()
        session.expunge(admin)
        return admin


def require_csrf(
    request: Request,
    admin: Annotated[Admin, Depends(require_admin)],
) -> Admin:
    expected = request.session.get(SESSION_CSRF_TOKEN)
    supplied = request.headers.get("X-CSRF-Token")
    expected_value = expected if isinstance(expected, str) else ""
    supplied_value = supplied if supplied is not None else ""
    if not hmac.compare_digest(expected_value, supplied_value) or not expected_value:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="CSRF validation failed",
        )
    return admin


AuthenticatedAdmin = Annotated[Admin, Depends(require_admin)]
CsrfAdmin = Annotated[Admin, Depends(require_csrf)]


def _unauthorized() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Unauthorized",
    )

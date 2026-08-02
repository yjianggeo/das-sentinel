#!/bin/sh
# Container entrypoint: "api" runs migrations then Uvicorn, "worker" starts
# the polling analysis worker, anything else is executed verbatim.
set -eu

case "${1:-api}" in
  api)
    alembic upgrade head
    exec uvicorn das_sentinel.api.app:app --host 0.0.0.0 --port 8000
    ;;
  worker)
    exec python -m das_sentinel.worker
    ;;
  *)
    exec "$@"
    ;;
esac

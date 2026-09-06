#!/bin/sh
# Container entrypoint: "api" runs migrations then Uvicorn, "worker" starts
# the polling analysis worker, anything else is executed verbatim.
set -eu

export DAS_DATABASE_URL="${DAS_DATABASE_URL:-sqlite+pysqlite:////app-data/das-sentinel.sqlite3}"

case "${1:-api}" in
  api)
    python -m das_sentinel.runtime
    alembic upgrade head
    exec uvicorn das_sentinel.api.app:create_runtime_app --factory --host 0.0.0.0 --port 8000
    ;;
  worker)
    exec python -m das_sentinel.worker
    ;;
  *)
    exec "$@"
    ;;
esac

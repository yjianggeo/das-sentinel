from __future__ import annotations

import os
from logging.config import fileConfig

from alembic import context
from das_sentinel.persistence.db import create_local_engine
from das_sentinel.persistence.models import Base

configuration = context.config
if configuration.config_file_name is not None:
    fileConfig(configuration.config_file_name)
target_metadata = Base.metadata


def _database_url() -> str:
    database_url = os.environ.get("DAS_DATABASE_URL") or configuration.get_main_option(
        "sqlalchemy.url"
    )
    if not database_url:
        raise RuntimeError("DAS_DATABASE_URL or sqlalchemy.url is required")
    return database_url


def run_migrations_offline() -> None:
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    engine = create_local_engine(_database_url())
    try:
        with engine.connect() as connection:
            context.configure(
                connection=connection,
                target_metadata=target_metadata,
                render_as_batch=True,
            )
            with context.begin_transaction():
                context.run_migrations()
    finally:
        engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

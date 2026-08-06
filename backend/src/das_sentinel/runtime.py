from pathlib import Path
from tempfile import NamedTemporaryFile

from das_sentinel.config import Settings
from das_sentinel.persistence.db import validate_selected_sqlite_runtime


class RuntimeProbeError(RuntimeError):
    pass


def _require_readable_directory(path: Path) -> None:
    if not path.is_dir():
        raise RuntimeProbeError("required directory is unavailable")
    next(path.iterdir(), None)


def probe_storage(settings: Settings) -> None:
    try:
        _require_readable_directory(settings.data_root)
        artifact_root = settings.app_data_dir / "artifacts"
        artifact_root.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(
            dir=artifact_root,
            prefix=".health-",
            delete=True,
        ) as probe:
            probe.write(b"ok")
            probe.flush()
    except OSError as error:
        raise RuntimeProbeError("required storage is unavailable") from error


def preflight_from_environment() -> Settings:
    settings = Settings()  # type: ignore[call-arg]
    validate_selected_sqlite_runtime()
    return settings


def main() -> None:
    preflight_from_environment()


if __name__ == "__main__":
    main()

import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile


def test_wheel_contains_database_migrations(tmp_path: Path) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            ".",
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            str(tmp_path),
        ],
        check=True,
        cwd=repository_root,
    )

    wheel_path = next(tmp_path.glob("das_sentinel-*.whl"))
    with ZipFile(wheel_path) as wheel:
        contents = set(wheel.namelist())

    assert {
        "das_sentinel/persistence/migrations/env.py",
        "das_sentinel/persistence/migrations/versions/0001_initial.py",
        "das_sentinel/persistence/migrations/versions/0002_job_model_and_review_request.py",
    } <= contents

# DAS Sentinel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a new NAS-ready DAS event platform that reads ZD/PRODML HDF5, detects and classifies earthquake/blast/traffic/noise events with multiple explainable methods, renders waterfall and spectrum products, supports review-driven training, and runs on CPU or NVIDIA CUDA.

**Architecture:** A modular Python monolith exposes one domain layer to FastAPI, a polling worker, and a CLI. The API is the only SQLite writer; workers read raw HDF5 and submit versioned results through an internal API. A React/TypeScript single-page app is built into the API image, while CPU and CUDA workers use separate Compose overrides.

**Tech Stack:** Python 3.12, FastAPI, Pydantic, SQLAlchemy 2, Alembic, NumPy, SciPy, h5py, scikit-learn, Matplotlib, optional CuPy CUDA 12, React 19, TypeScript, Vite, Vitest, Testing Library, Playwright, Docker Compose.

## Global Constraints

- Work only in the new `Z:\das-event-platform` repository; do not import code from or modify `Z:\das-events-work\das-events`.
- Raw DAS mounts are read-only; no implementation may move, rename, truncate, or delete source files.
- Canonical arrays are `float32` with shape `[time, channel]`; internal timestamps are timezone-aware UTC.
- First reader supports `Acquisition/Raw[0]/RawData` and `Acquisition/Raw[0]/RawDataTime`; other formats use the reader protocol.
- Labels are exactly `earthquake`, `blast`, `traffic`, `noise`; `unknown` is fusion abstention and is never a supervised training class.
- Default processing blocks are 20 seconds with 5 seconds overlap.
- CPU and CUDA paths must agree within `rtol=1e-4, atol=1e-5` for shared numerical kernels.
- NAS default worker concurrency is 1; GPU default worker concurrency is 1; worker prefetch is 1 file.
- SQLite lives on a deployment-host local volume, uses WAL, has one application-level writer, and must be SQLite 3.51.3+ or a documented patched release.
- API runs as one Uvicorn process; the React static build is served by that process.
- A published site model requires at least 20 reviewed events in each of the four supervised classes, grouped validation macro-F1 >= 0.70, and every class recall >= 0.50.
- If CUDA is absent or runs out of memory, record the failure and retry the file once on CPU.
- Every persisted run records site profile version, pipeline version, model version, source file fingerprint, and actual compute backend.
- Use test-driven development: observe the focused test fail before writing implementation.
- Preserve the repository-local Git identity and do not modify global Git configuration.

## Scope Decision

The system is broad but the requested deliverable is one deployable application. Signal
types feed the shared event contract; persistence and worker protocol consume that same
contract; the Web UI consumes those APIs; model training consumes reviewed event features.
Splitting these into independent plans would duplicate or prematurely freeze interfaces.
The 22 tasks below therefore remain one ordered plan, with independently testable commits
at the core-library, API/worker, Web, deployment, and acceptance milestones.

## File Map

```text
pyproject.toml                          Python package, dependencies, test/lint config
backend/src/das_sentinel/
  config.py                             Settings and immutable processing profiles
  domain.py                             Canonical blocks, candidates, events, labels
  readers/base.py                       Reader protocol and registry
  readers/zd_hdf5.py                    ZD/PRODML HDF5 reader
  signal/backend.py                     NumPy/CuPy selection and CPU fallback
  signal/preprocess.py                  Validation, bad channels, detrend, FFT band-pass
  detectors/stalta.py                   STA/LTA channel-coincidence detector
  detectors/energy.py                   Multiband statistical detector
  detectors/coherence.py                Debiased slant-stack coherence detector
  detectors/trajectory.py               Traffic trajectory and event-template scores
  fusion.py                             Candidate gating, merging, stable event IDs
  features.py                           Event feature vector extraction
  classifiers/rules.py                  Explainable physical rule scorer
  classifiers/similarity.py             HDBSCAN and nearest reviewed events
  classifiers/supervised.py             Random-forest training and validation
  classifiers/fusion.py                 Rule/model fusion and unknown abstention
  artifacts.py                          Waterfall, waveform, Welch PSD, STFT
  pipeline.py                           Pure single-file processing orchestration
  persistence/models.py                 SQLAlchemy entities
  persistence/db.py                     Engine, sessions, WAL/version checks
  persistence/repositories.py           Typed persistence operations
  api/app.py                            FastAPI application and SPA serving
  api/auth.py                           Single-admin session and CSRF
  api/routes/*.py                       Sites, jobs, events, models, internal worker API
  services/discovery.py                 Batch discovery and stable-file polling
  services/jobs.py                      Atomic claim, heartbeat, retry, recovery
  services/training.py                  Dataset snapshot and model lifecycle
  worker/main.py                        Poll, process, submit, fallback
  cli.py                                `das-sentinel` commands
frontend/src/
  api.ts                                Typed same-origin API client
  auth.tsx                              Session state
  App.tsx                               Router and shell
  pages/*.tsx                           Login, dashboard, sites, jobs, events, models
  components/*.tsx                      Event list, artifact panes, evidence, forms
docker/
  Dockerfile.cpu                        API and CPU worker image
  Dockerfile.gpu                        CUDA worker image
compose.yaml                            API and persistent application volume
compose.cpu.yaml                        CPU worker override
compose.gpu.yaml                        NVIDIA worker override
tests/                                  Python unit/integration/performance tests
frontend/src/**/*.test.tsx              Frontend component tests
e2e/                                   Playwright system tests
```

---

### Task 1: Python foundation, settings, and domain contracts

**Files:**
- Create: `pyproject.toml`
- Create: `backend/src/das_sentinel/__init__.py`
- Create: `backend/src/das_sentinel/config.py`
- Create: `backend/src/das_sentinel/domain.py`
- Create: `tests/test_config.py`
- Create: `tests/test_domain.py`

**Interfaces:**
- Produces: `Settings`, `ProcessingProfile`, `DasMetadata`, `DasBlock`, `Candidate`, `EventRecord`, `Label`, `ComputeBackend`.
- `ProcessingProfile.model_copy(update=...)` is the supported way tests and later tasks derive profiles.

- [ ] **Step 1: Write failing profile and domain tests**

```python
# tests/test_config.py
import pytest
from pydantic import ValidationError
from das_sentinel.config import ProcessingProfile

def test_profile_has_approved_block_defaults():
    profile = ProcessingProfile()
    assert profile.block_seconds == 20.0
    assert profile.overlap_seconds == 5.0
    assert profile.device == "auto"

def test_profile_rejects_overlap_not_smaller_than_block():
    with pytest.raises(ValidationError):
        ProcessingProfile(block_seconds=5.0, overlap_seconds=5.0)

# tests/test_domain.py
from datetime import datetime, timezone
import numpy as np
import pytest
from das_sentinel.domain import DasBlock, DasMetadata

def test_das_block_requires_time_channel_float32():
    meta = DasMetadata(
        source_path="/data/a.h5", start_time_utc=datetime.now(timezone.utc),
        sample_rate_hz=1000.0, channel_distance_m=np.arange(3, dtype=np.float32),
        unit="strain_rate",
    )
    with pytest.raises(ValueError, match="float32"):
        DasBlock(data=np.zeros((10, 3), dtype=np.float64), metadata=meta, sample_offset=0)
```

- [ ] **Step 2: Run tests and observe the missing-package failure**

Run: `python -m pytest tests/test_config.py tests/test_domain.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'das_sentinel'`.

- [ ] **Step 3: Add package metadata and exact public contracts**

```toml
# pyproject.toml
[build-system]
requires = ["setuptools>=75"]
build-backend = "setuptools.build_meta"

[project]
name = "das-sentinel"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
  "fastapi>=0.115",
  "uvicorn[standard]>=0.34",
  "pydantic>=2.10",
  "pydantic-settings>=2.7",
  "sqlalchemy>=2.0",
  "alembic>=1.14",
  "numpy>=2.0",
  "scipy>=1.14",
  "h5py>=3.12",
  "scikit-learn>=1.6",
  "matplotlib>=3.9",
  "pysqlite3-binary>=0.5.4; platform_system == 'Linux' and platform_machine == 'x86_64'",
  "argon2-cffi>=23.1",
  "itsdangerous>=2.2",
  "httpx>=0.28",
  "typer>=0.15",
]

[project.optional-dependencies]
gpu = ["cupy-cuda12x>=14.0"]
dev = ["pytest>=8.3", "pytest-cov>=6", "ruff>=0.9", "mypy>=1.14", "PyYAML>=6.0"]

[project.scripts]
das-sentinel = "das_sentinel.cli:app"

[tool.setuptools.packages.find]
where = ["backend/src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["backend/src"]
```

```python
# backend/src/das_sentinel/domain.py
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any
import numpy as np

class Label(StrEnum):
    EARTHQUAKE = "earthquake"
    BLAST = "blast"
    TRAFFIC = "traffic"
    NOISE = "noise"
    UNKNOWN = "unknown"

class ComputeBackend(StrEnum):
    CPU = "cpu"
    CUDA = "cuda"

@dataclass(frozen=True)
class DasMetadata:
    source_path: str
    start_time_utc: datetime
    sample_rate_hz: float
    channel_distance_m: np.ndarray
    unit: str
    warnings: tuple[str, ...] = ()

@dataclass(frozen=True)
class DasBlock:
    data: np.ndarray
    metadata: DasMetadata
    sample_offset: int
    def __post_init__(self) -> None:
        if self.data.dtype != np.float32:
            raise ValueError("DasBlock data must be float32")
        if self.data.ndim != 2:
            raise ValueError("DasBlock data must have shape [time, channel]")
        if self.data.shape[1] != len(self.metadata.channel_distance_m):
            raise ValueError("channel distance count does not match data")

@dataclass(frozen=True)
class Candidate:
    detector: str
    start_sample: int
    end_sample: int
    peak_sample: int
    channel_start: int
    channel_stop: int
    score: float
    metrics: dict[str, float] = field(default_factory=dict)
    quality_flags: tuple[str, ...] = ()

@dataclass(frozen=True)
class EventRecord:
    event_id: str
    start_sample: int
    end_sample: int
    peak_sample: int
    channel_start: int
    channel_stop: int
    evidence: tuple[Candidate, ...]
    features: dict[str, float] = field(default_factory=dict)
    label: Label = Label.UNKNOWN
    decision_score: float = 0.0
    reasons: tuple[str, ...] = ()
```

Implement `ProcessingProfile` as a frozen Pydantic model with defaults copied from this plan and a model-level validator enforcing positive sampling windows, valid frequency order, `overlap_seconds < block_seconds`, and `device in {"auto","cpu","cuda"}`. Implement `Settings` with `data_root`, `app_data_dir`, `database_url`, `session_secret`, `worker_token`, `secure_cookies`, and `pipeline_version`.

```python
# backend/src/das_sentinel/config.py
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class ProcessingProfile(BaseModel):
    model_config = ConfigDict(frozen=True)
    block_seconds: float = 20.0
    overlap_seconds: float = 5.0
    device: Literal["auto", "cpu", "cuda"] = "auto"
    common_mode_removal: bool = True
    bad_channel_rms_z: float = 8.0
    stalta_freq_low_hz: float = 2.0
    stalta_freq_high_hz: float = 40.0
    sta_seconds: float = 0.5
    lta_seconds: float = 10.0
    stalta_threshold: float = 4.0
    stalta_strong_ratio: float = 8.0
    min_coincidence: int = 4
    min_duration_seconds: float = 0.2
    energy_bands_hz: tuple[tuple[float, float], ...] = (
        (0.5, 5.0), (5.0, 20.0), (20.0, 80.0),
    )
    energy_window_seconds: float = 2.0
    energy_ratio_threshold: float = 4.0
    coherence_freq_low_hz: float = 1.0
    coherence_freq_high_hz: float = 40.0
    coherence_window_seconds: float = 2.0
    coherence_threshold: float = 0.08
    slowness_max_s_per_m: float = 0.002
    slowness_points: int = 21
    trajectory_freq_low_hz: float = 1.0
    trajectory_freq_high_hz: float = 30.0
    trajectory_velocity_min_mps: float = 2.0
    trajectory_velocity_max_mps: float = 60.0
    trajectory_threshold: float = 0.65
    template_threshold: float = 0.75
    fusion_normal_score: float = 0.60
    fusion_strong_score: float = 0.85
    merge_gap_seconds: float = 1.0
    artifact_pad_seconds: float = 5.0
    recursive_discovery: bool = True

    @model_validator(mode="after")
    def validate_windows_and_bands(self) -> "ProcessingProfile":
        if self.block_seconds <= 0 or not 0 <= self.overlap_seconds < self.block_seconds:
            raise ValueError("overlap_seconds must be nonnegative and smaller than block_seconds")
        if self.sta_seconds <= 0 or self.lta_seconds <= self.sta_seconds:
            raise ValueError("lta_seconds must be greater than sta_seconds")
        if self.slowness_points < 3 or self.slowness_points % 2 == 0:
            raise ValueError("slowness_points must be odd and at least 3")
        for low, high in self.energy_bands_hz:
            if low <= 0 or high <= low:
                raise ValueError("every frequency band must satisfy 0 < low < high")
        return self

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DAS_", env_file=".env")
    data_root: Path = Path("/data")
    app_data_dir: Path = Path("/app-data")
    database_url: str = "sqlite+pysqlite:////app-data/das-sentinel.sqlite3"
    api_url: str = "http://127.0.0.1:8000"
    session_secret: str = Field(min_length=32)
    worker_token: str = Field(min_length=32)
    secure_cookies: bool = True
    pipeline_version: str = "v1"
    worker_device: Literal["auto", "cpu", "cuda"] = "auto"
```

- [ ] **Step 4: Run focused tests**

Run: `python -m pytest tests/test_config.py tests/test_domain.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml backend/src/das_sentinel tests/test_config.py tests/test_domain.py
git commit -m "feat(core): add settings and DAS domain contracts"
```

---

### Task 2: Reader registry and ZD/PRODML HDF5

**Files:**
- Create: `backend/src/das_sentinel/readers/__init__.py`
- Create: `backend/src/das_sentinel/readers/base.py`
- Create: `backend/src/das_sentinel/readers/zd_hdf5.py`
- Create: `tests/fixtures.py`
- Create: `tests/test_zd_hdf5.py`

**Interfaces:**
- Consumes: `DasMetadata`, `DasBlock`.
- Produces: `DasReader` protocol, `ReaderRegistry`,
  `ZdHdf5Reader.probe/read_metadata/sample_count/read_block`.

- [ ] **Step 1: Add a synthetic HDF5 fixture and failing reader tests**

```python
# tests/fixtures.py
from datetime import datetime, timezone
import h5py
import numpy as np

def write_zd_h5(path, *, n_time=200, n_channel=8, fs=100.0):
    start = datetime(2026, 7, 26, tzinfo=timezone.utc)
    times = start.timestamp() * 1_000_000 + np.arange(n_time) * 1_000_000 / fs
    with h5py.File(path, "w") as handle:
        raw = handle.create_group("Acquisition").create_group("Raw[0]")
        raw.create_dataset("RawData", data=np.zeros((n_time, n_channel), np.float32))
        raw.create_dataset("RawDataTime", data=times.astype(np.int64))
        raw.attrs["OutputDataRate"] = fs
        raw.attrs["SpatialSamplingInterval"] = 4.0
        raw.attrs["StartLocusIndex"] = 2
        raw.attrs["RawDataUnit"] = "strain_rate"

# tests/test_zd_hdf5.py
import numpy as np
from das_sentinel.readers.zd_hdf5 import ZdHdf5Reader
from tests.fixtures import write_zd_h5

def test_reads_metadata_and_slice(tmp_path):
    path = tmp_path / "sample.h5"
    write_zd_h5(path)
    reader = ZdHdf5Reader()
    assert reader.probe(path).supported
    block = reader.read_block(path, 10, 30, slice(1, 4))
    assert block.data.shape == (20, 3)
    assert block.data.dtype == np.float32
    np.testing.assert_allclose(block.metadata.channel_distance_m, [12, 16, 20])
    assert block.metadata.start_time_utc.isoformat() == "2026-07-26T00:00:00.100000+00:00"
```

- [ ] **Step 2: Verify failure**

Run: `python -m pytest tests/test_zd_hdf5.py -q`

Expected: FAIL importing `das_sentinel.readers.zd_hdf5`.

- [ ] **Step 3: Implement the reader protocol and strict HDF5 normalization**

```python
# backend/src/das_sentinel/readers/base.py
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from das_sentinel.domain import DasBlock, DasMetadata

@dataclass(frozen=True)
class ProbeResult:
    supported: bool
    format_name: str
    reason: str = ""

class DasReader(Protocol):
    def probe(self, path: Path) -> ProbeResult: ...
    def read_metadata(self, path: Path) -> DasMetadata: ...
    def sample_count(self, path: Path) -> int: ...
    def read_block(self, path: Path, sample_start: int, sample_stop: int,
                   channel_slice: slice = slice(None)) -> DasBlock: ...

class ReaderRegistry:
    def __init__(self, readers: list[DasReader]) -> None:
        self._readers = readers
    def resolve(self, path: Path) -> DasReader:
        for reader in self._readers:
            if reader.probe(path).supported:
                return reader
        raise ValueError(f"No DAS reader supports {path}")
```

Implement `ZdHdf5Reader` with exact dataset constants, inclusive validation of sample/channel bounds, orientation detection from `RawDataTime` length, microsecond UTC conversion, channel distance calculation `(StartLocusIndex + index) * SpatialSamplingInterval`, and warnings when filename time disagrees with data time. Do not load a full dataset in `read_metadata`.

- [ ] **Step 4: Run reader tests**

Run: `python -m pytest tests/test_zd_hdf5.py -q`

Expected: PASS, including added tests for transposed data, missing dataset, and invalid bounds.

- [ ] **Step 5: Commit**

```bash
git add backend/src/das_sentinel/readers tests/fixtures.py tests/test_zd_hdf5.py
git commit -m "feat(io): add plugin reader and ZD HDF5 support"
```

---

### Task 3: CPU/CUDA array backend and preprocessing

**Files:**
- Create: `backend/src/das_sentinel/signal/__init__.py`
- Create: `backend/src/das_sentinel/signal/backend.py`
- Create: `backend/src/das_sentinel/signal/preprocess.py`
- Create: `tests/test_backend.py`
- Create: `tests/test_preprocess.py`

**Interfaces:**
- Produces: `ArrayBackend`, `select_backend(device)`, `preprocess_block(block, profile, backend) -> PreprocessedBlock`.
- `PreprocessedBlock.data` remains on the selected array backend; `bad_channel_mask` is a CPU boolean array.

- [ ] **Step 1: Write failing CPU and fallback tests**

```python
import numpy as np
from das_sentinel.signal.backend import select_backend
from das_sentinel.signal.preprocess import fft_bandpass, find_bad_channels

def test_cpu_fft_bandpass_preserves_ten_hz_and_removes_dc():
    fs = 100.0
    t = np.arange(1000) / fs
    x = (4.0 + np.sin(2 * np.pi * 10 * t))[:, None].astype(np.float32)
    y = fft_bandpass(x, fs, 5.0, 20.0, select_backend("cpu"))
    assert abs(float(y.mean())) < 1e-3
    assert np.corrcoef(y[:, 0], np.sin(2 * np.pi * 10 * t))[0, 1] > 0.99

def test_constant_and_nonfinite_channels_are_bad():
    x = np.random.default_rng(4).normal(size=(100, 4)).astype(np.float32)
    x[:, 1] = 0
    x[2, 3] = np.nan
    assert find_bad_channels(x).tolist() == [False, True, False, True]

def test_auto_backend_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr("das_sentinel.signal.backend._load_cupy", lambda: None)
    backend = select_backend("auto")
    assert backend.name == "cpu"
```

- [ ] **Step 2: Verify failure**

Run: `python -m pytest tests/test_backend.py tests/test_preprocess.py -q`

Expected: FAIL because `das_sentinel.signal` does not exist.

- [ ] **Step 3: Implement backend selection and deterministic preprocessing**

`ArrayBackend` exposes `name`, `xp`, `asarray`, `asnumpy`, and `clear_cache`.
`select_backend("cuda")` raises `BackendUnavailable`; `"auto"` records the CUDA load
error and returns NumPy. Implement FFT band-pass using the same raised-cosine frequency
gain on NumPy and CuPy so parity is testable. Implement finite checks, constant/RMS bad
channel detection, linear detrend, optional per-time channel median removal, and robust
median/MAD scaling. Never mutate the input `DasBlock`.

```python
@dataclass(frozen=True)
class PreprocessedBlock:
    data: Any
    bad_channel_mask: np.ndarray
    backend_name: str
    sample_offset: int
    sample_rate_hz: float
    channel_distance_m: np.ndarray
```

- [ ] **Step 4: Verify CPU behavior and optional CUDA parity**

Run: `python -m pytest tests/test_backend.py tests/test_preprocess.py -q`

Expected: PASS. If CUDA is available, `test_cuda_fft_matches_cpu` must run with
`np.testing.assert_allclose(..., rtol=1e-4, atol=1e-5)`; otherwise it must SKIP with
the recorded unavailable reason.

- [ ] **Step 5: Commit**

```bash
git add backend/src/das_sentinel/signal tests/test_backend.py tests/test_preprocess.py
git commit -m "feat(signal): add CPU CUDA backend and preprocessing"
```

---

### Task 4: STA/LTA and multiband statistical detectors

**Files:**
- Create: `backend/src/das_sentinel/detectors/__init__.py`
- Create: `backend/src/das_sentinel/detectors/stalta.py`
- Create: `backend/src/das_sentinel/detectors/energy.py`
- Create: `tests/synthetic.py`
- Create: `tests/test_stalta.py`
- Create: `tests/test_energy.py`

**Interfaces:**
- Consumes: `PreprocessedBlock`, `ProcessingProfile`.
- Produces: `detect_stalta(...) -> list[Candidate]`, `detect_energy(...) -> list[Candidate]`.

- [ ] **Step 1: Write deterministic positive and negative detector tests**

```python
def test_stalta_finds_multichannel_impulse(preprocessed_impulse, profile):
    candidates = detect_stalta(preprocessed_impulse, profile)
    assert len(candidates) == 1
    assert 490 <= candidates[0].peak_sample <= 510
    assert candidates[0].metrics["peak_ratio"] >= profile.stalta_threshold
    assert 0.0 <= candidates[0].score <= 1.0

def test_stalta_rejects_single_channel_impulse(preprocessed_single_channel, profile):
    assert detect_stalta(preprocessed_single_channel, profile) == []

def test_energy_finds_sustained_band_limited_event(preprocessed_sustained, profile):
    candidates = detect_energy(preprocessed_sustained, profile)
    assert candidates[0].metrics["spectral_entropy"] < 0.9
    assert candidates[0].end_sample - candidates[0].start_sample >= 200
```

Create synthetic fixtures with fixed RNG seeds and explicit injected time/channel spans.

- [ ] **Step 2: Verify failure**

Run: `python -m pytest tests/test_stalta.py tests/test_energy.py -q`

Expected: FAIL importing detector modules.

- [ ] **Step 3: Implement detector-specific evidence**

STA/LTA uses squared signal, cumulative-window means, a safe epsilon, channel coincidence,
and run grouping. Its trailing windows use only past samples: the leading LTA eligibility
already supplies edge protection, so do not discard a second full LTA window at the right
edge. A default 20-second block with a 10-second LTA must retain enough eligible samples
for the configured minimum event duration. Energy detection computes window RMS ratio,
excess kurtosis, peak/median, normalized Welch band powers, and spectral entropy. Each
detector returns global sample indexes by adding
`PreprocessedBlock.sample_offset`. For an accepted raw ratio `r` and detector threshold
`t`, normalize the candidate score as `0.60 + 0.40 * clip((r - t) / t, 0, 1)`;
retain the unmodified raw ratio in candidate metrics.

- [ ] **Step 4: Run focused and regression tests**

Run: `python -m pytest tests/test_stalta.py tests/test_energy.py tests/test_preprocess.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/src/das_sentinel/detectors tests/synthetic.py tests/test_stalta.py tests/test_energy.py
git commit -m "feat(detect): add transient and multiband detectors"
```

---

### Task 5: Debiased spatial coherence and moveout

**Files:**
- Create: `backend/src/das_sentinel/detectors/coherence.py`
- Create: `tests/test_coherence.py`

**Interfaces:**
- Produces: `detect_coherence(block, profile, backend) -> list[Candidate]`.
- Candidate metrics include `semblance`, `slowness_s_per_m`, `apparent_velocity_mps`, and `direction`.

- [ ] **Step 1: Add failing moveout and common-mode tests**

```python
def test_coherence_recovers_moving_wavefront(moving_wavefront, profile, cpu_backend):
    candidate = detect_coherence(moving_wavefront, profile, cpu_backend)[0]
    assert candidate.metrics["semblance"] > 0.1
    assert candidate.metrics["slowness_s_per_m"] == pytest.approx(2.5e-4, abs=5e-5)
    assert candidate.metrics["direction"] == 1.0

def test_coherence_flags_zero_slowness_common_mode(common_mode_pulse, profile, cpu_backend):
    candidates = detect_coherence(common_mode_pulse, profile, cpu_backend)
    assert candidates == [] or "zero_slowness_common_mode" in candidates[0].quality_flags
```

- [ ] **Step 2: Verify failure**

Run: `python -m pytest tests/test_coherence.py -q`

Expected: FAIL importing `detect_coherence`.

- [ ] **Step 3: Implement slant-stack scanning**

Build an odd slowness grid containing zero, convert each distance delay to integer samples,
shift valid channel traces, and compute squared stack energy divided by summed individual
energy. Subtract the finite-channel noise expectation `1 / n_valid`, clip to `[0, 1]`, and
retain the best slowness. For accepted semblance `s` and configured threshold `t`, normalize
the candidate score as `0.60 + 0.40 * clip((s - t) / t, 0, 1)`. The NumPy/CuPy kernel must
share one implementation through `ArrayBackend.xp`.

- [ ] **Step 4: Verify behavior and backend parity**

Run: `python -m pytest tests/test_coherence.py tests/test_backend.py -q`

Expected: PASS; CUDA parity SKIPs only when no CUDA runtime is present.

- [ ] **Step 5: Commit**

```bash
git add backend/src/das_sentinel/detectors/coherence.py tests/test_coherence.py
git commit -m "feat(detect): add debiased coherence and moveout"
```

---

### Task 6: Traffic trajectories and reviewed-event templates

**Files:**
- Create: `backend/src/das_sentinel/detectors/trajectory.py`
- Create: `tests/test_trajectory.py`

**Interfaces:**
- Produces: `detect_trajectories(block, profile)`, `Template`, `score_template(block, template)`.

- [ ] **Step 1: Add failing line-trajectory and template tests**

```python
def test_trajectory_estimates_vehicle_speed(vehicle_diagonal, profile):
    candidate = detect_trajectories(vehicle_diagonal, profile)[0]
    assert candidate.detector == "trajectory"
    assert candidate.metrics["apparent_velocity_mps"] == pytest.approx(18.0, rel=0.2)

def test_template_score_prefers_matching_pulse_train(blast_block, unrelated_block):
    template = Template.from_event(blast_block, event_id="reviewed-blast")
    assert score_template(blast_block, template) > 0.8
    assert score_template(unrelated_block, template) < 0.4
```

- [ ] **Step 2: Verify failure**

Run: `python -m pytest tests/test_trajectory.py -q`

Expected: FAIL importing trajectory contracts.

- [ ] **Step 3: Implement bounded Hough scoring and normalized templates**

Compute a low-frequency energy image on a decimated time/distance grid. Scan configured
velocity bounds by summing energy along candidate lines, apply non-maximum suppression,
and map detections back to full sample/channel indexes. Trajectory and normalized
cross-correlation template scores are already in `[0, 1]`; only scores at or above their
configured detector threshold become candidates. Store templates as normalized
beam waveform plus spectrum, site ID, profile version, source event ID, and sampling
metadata; reject templates from mismatched site/profile.

- [ ] **Step 4: Run trajectory tests**

Run: `python -m pytest tests/test_trajectory.py -q`

Expected: PASS, including a horizontal common-mode rejection test.

- [ ] **Step 5: Commit**

```bash
git add backend/src/das_sentinel/detectors/trajectory.py tests/test_trajectory.py
git commit -m "feat(detect): add traffic trajectories and templates"
```

---

### Task 7: Balanced candidate fusion and stable events

**Files:**
- Create: `backend/src/das_sentinel/fusion.py`
- Create: `tests/test_fusion.py`

**Interfaces:**
- Produces:
  `fuse_candidates(candidates, profile, sample_rate_hz, site_id, pipeline_version, source_fingerprint=None) -> list[EventRecord]`.
- `sample_rate_hz` is explicit so `merge_gap_seconds` is converted to samples without
  hiding structural timing data inside detector metrics.
- `source_fingerprint` is optional for backward-compatible detector-unit use. File
  pipelines must provide it so two source files with the same sample/channel coordinates
  cannot generate the same event UUID.

- [ ] **Step 1: Write failing balanced-gate tests**

```python
def test_two_overlapping_normal_candidates_form_event(profile):
    events = fuse_candidates(
        [candidate("stalta", 100, 180, 0.65), candidate("energy", 120, 190, 0.64)],
        profile, 100.0, "site-a", "v1",
    )
    assert len(events) == 1
    assert {e.detector for e in events[0].evidence} == {"stalta", "energy"}

def test_single_weak_candidate_stays_out_of_review_queue(profile):
    assert fuse_candidates(
        [candidate("energy", 100, 180, 0.45)], profile, 100.0, "site-a", "v1"
    ) == []

def test_event_id_is_deterministic(profile):
    first = fuse_candidates([candidate("stalta", 100, 180, 0.95)], profile, 100.0, "s", "v1")[0]
    second = fuse_candidates([candidate("stalta", 100, 180, 0.95)], profile, 100.0, "s", "v1")[0]
    assert first.event_id == second.event_id
```

- [ ] **Step 2: Verify failure**

Run: `python -m pytest tests/test_fusion.py -q`

Expected: FAIL importing `fuse_candidates`.

- [ ] **Step 3: Implement interval union, evidence gates, and UUID5 IDs**

Use temporal overlap plus channel-range overlap, allow the profile merge gap, merge across
block boundaries, and require either two ordinary detectors or one strong, quality-valid
detector. Generate UUID5 from site ID, pipeline version, optional source fingerprint,
merged peak sample, and merged channel interval. Preserve every evidence object sorted by
detector and peak.

- [ ] **Step 4: Run fusion tests**

Run: `python -m pytest tests/test_fusion.py -q`

Expected: PASS, including cross-file-offset and zero-slowness veto cases.

- [ ] **Step 5: Commit**

```bash
git add backend/src/das_sentinel/fusion.py tests/test_fusion.py
git commit -m "feat(core): fuse candidates into stable events"
```

---

### Task 8: Feature extraction, rule classification, and abstention

**Files:**
- Create: `backend/src/das_sentinel/features.py`
- Create: `backend/src/das_sentinel/classifiers/__init__.py`
- Create: `backend/src/das_sentinel/classifiers/rules.py`
- Create: `backend/src/das_sentinel/classifiers/fusion.py`
- Create: `tests/test_features.py`
- Create: `tests/test_rule_classifier.py`
- Create: `tests/test_classification_fusion.py`

**Interfaces:**
- Produces: `extract_features`, `RuleResult`, `score_rules`, `fuse_classification`.

- [ ] **Step 1: Add failing feature and classification tests**

```python
def test_extracts_frequency_duration_and_spatial_features(event_block, event):
    features = extract_features(event_block, event, fs=100.0)
    assert features["duration_s"] == pytest.approx(2.0)
    assert features["dominant_frequency_hz"] == pytest.approx(12.0, rel=0.1)
    assert features["distance_extent_m"] > 100.0

def test_rule_classifier_explains_traffic(traffic_features, site_profile):
    result = score_rules(traffic_features, site_profile)
    assert max(result.scores, key=result.scores.get) == Label.TRAFFIC
    assert "trajectory_speed_in_range" in result.reasons[Label.TRAFFIC]

def test_fusion_abstains_when_top_scores_are_close():
    result = fuse_classification(
        rule_scores={Label.EARTHQUAKE: 0.62, Label.BLAST: 0.55},
        model_probabilities=None,
        quality_flags=(),
    )
    assert result.label == Label.UNKNOWN
```

- [ ] **Step 2: Verify failure**

Run: `python -m pytest tests/test_features.py tests/test_rule_classifier.py tests/test_classification_fusion.py -q`

Expected: FAIL importing feature/classifier modules.

- [ ] **Step 3: Implement the approved feature schema and rule evidence**

Extract the exact temporal, spectral, spatial, propagation, and statistical feature names
from design spec section 9.1. Rules return scores in `[0, 1]` plus per-label reason tuples.
Fusion uses rules alone before a model exists; with a model it uses `0.35 * rules +
0.65 * model`. Return `unknown` below 0.60 or with top-two margin below 0.15. Common-mode,
bad-channel-majority, and edge-only flags downgrade to `noise` or `unknown`.

- [ ] **Step 4: Run focused tests**

Run: `python -m pytest tests/test_features.py tests/test_rule_classifier.py tests/test_classification_fusion.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/src/das_sentinel/features.py backend/src/das_sentinel/classifiers tests/test_features.py tests/test_rule_classifier.py tests/test_classification_fusion.py
git commit -m "feat(classify): add explainable features rules and abstention"
```

---

### Task 9: Similarity index and supervised model lifecycle

**Files:**
- Create: `backend/src/das_sentinel/classifiers/similarity.py`
- Create: `backend/src/das_sentinel/classifiers/supervised.py`
- Create: `tests/test_similarity.py`
- Create: `tests/test_supervised.py`

**Interfaces:**
- Produces: `SimilarityIndex.fit/query`, `TrainingReport`, `train_model`, `save_model_bundle`, `load_model_bundle`.
- `save_model_bundle` returns the SHA-256 of the complete saved bundle.
- `load_model_bundle` requires an `expected_sha256` obtained from trusted persistence and
  verifies it before parsing the outer ZIP, JSON metadata, or skops model payload.

- [ ] **Step 1: Add failing semantic and release-gate tests**

```python
def test_similarity_returns_cluster_and_reviewed_neighbors(feature_rows):
    index = SimilarityIndex().fit(feature_rows)
    result = index.query(feature_rows[0].features, limit=3)
    assert result.cluster_id is not None
    assert result.neighbors[0].event_id == feature_rows[0].event_id

def test_training_rejects_less_than_twenty_per_class(training_rows):
    with pytest.raises(TrainingGateError, match="20 reviewed events"):
        train_model(training_rows[:60], groups=[row.day for row in training_rows[:60]])

def test_publishable_report_meets_all_gates(full_training_rows):
    bundle, report = train_model(full_training_rows, groups=[r.day for r in full_training_rows])
    assert report.macro_f1 >= 0.70
    assert min(report.recall_by_label.values()) >= 0.50
    assert bundle.feature_names == tuple(sorted(full_training_rows[0].features))
```

- [ ] **Step 2: Verify failure**

Run: `python -m pytest tests/test_similarity.py tests/test_supervised.py -q`

Expected: FAIL importing similarity/supervised modules.

- [ ] **Step 3: Implement deterministic sklearn pipelines**

Use `StandardScaler` for HDBSCAN/nearest-neighbor features and never map cluster IDs to
labels. Use `RandomForestClassifier(n_estimators=300, class_weight="balanced",
random_state=20260726, n_jobs=1)`. Validate with `StratifiedGroupKFold(n_splits=5,
shuffle=True, random_state=20260726)`. Serialize a bundle containing model, ordered
feature names, labels, training snapshot hash, sklearn version, and report; reject loads
whose feature schema differs.

- [ ] **Step 4: Run model tests**

Run: `python -m pytest tests/test_similarity.py tests/test_supervised.py -q`

Expected: PASS with deterministic metrics on separable fixtures.

- [ ] **Step 5: Commit**

```bash
git add backend/src/das_sentinel/classifiers tests/test_similarity.py tests/test_supervised.py
git commit -m "feat(ml): add similarity and supervised model gates"
```

---

### Task 10: Event artifact generation

**Files:**
- Create: `backend/src/das_sentinel/artifacts.py`
- Create: `tests/test_artifacts.py`

**Interfaces:**
- Produces: `EventArtifactData`, `ArtifactManifest`,
  `render_event_artifacts(data, event, features, out_dir)`.
- `EventArtifactData` explicitly carries distinct `raw_data` and `filtered_data` arrays,
  `sample_offset`, `sample_rate_hz`, `channel_distance_m`, `bad_channel_mask`,
  `start_time_utc`, IANA `site_timezone`, `profile`, explicit `profile_version`,
  `pipeline_version`, explicit `model_version`, `backend_name`, `rule_output`, and
  optional `model_output`. Use `rules-only` as `model_version` when no learned model
  participates. The two arrays must have the same `[time, channel]` shape; rendering must
  never relabel filtered data as raw.

- [ ] **Step 1: Add failing artifact tests**

```python
def test_renders_complete_event_artifact_set(tmp_path, event_artifact_data, event, features):
    manifest = render_event_artifacts(event_artifact_data, event, features, tmp_path)
    assert set(manifest.files) == {
        "waterfall_raw", "waterfall_filtered", "waveform", "psd", "stft", "evidence"
    }
    for artifact in manifest.files.values():
        assert artifact.path.exists()
        assert len(artifact.sha256) == 64
    evidence = json.loads(manifest.files["evidence"].path.read_text("utf-8"))
    assert evidence["event_id"] == event.event_id
```

- [ ] **Step 2: Verify failure**

Run: `python -m pytest tests/test_artifacts.py -q`

Expected: FAIL importing `render_event_artifacts`.

- [ ] **Step 3: Implement noninteractive, versioned rendering**

```python
@dataclass(frozen=True)
class ArtifactFile:
    kind: str
    path: Path
    media_type: str
    sha256: str

@dataclass(frozen=True)
class ArtifactManifest:
    event_id: str
    files: dict[str, ArtifactFile]
```

Use Matplotlib `Agg`, robust 1st/99th percentile limits, explicit UTC/local axes, physical
distance axes, `scipy.signal.welch`, and `scipy.signal.stft`. Write to a temporary sibling
file and atomically replace the final file while serializing writers across processes.
Derive directories from validated event UUIDs, not user strings. Every plot displays the
profile, pipeline, and model versions. Evidence JSON includes those versions, profile,
backend, source range, detections, features, rule/model outputs, and quality flags.

- [ ] **Step 4: Run artifact tests**

Run: `python -m pytest tests/test_artifacts.py -q`

Expected: PASS and no Matplotlib windows.

- [ ] **Step 5: Commit**

```bash
git add backend/src/das_sentinel/artifacts.py tests/test_artifacts.py
git commit -m "feat(artifacts): render waterfall and spectral products"
```

---

### Task 11: Pure single-file pipeline and benchmark

**Files:**
- Create: `backend/src/das_sentinel/pipeline.py`
- Create: `tests/test_pipeline.py`
- Create: `tests/test_performance.py`

**Interfaces:**
- Produces: `AnalysisContext`, `FileAnalysis`,
  `analyze_file(path, reader, profile, backend, artifact_root, context=None)`.
- `AnalysisContext` explicitly carries the stable run identity and site metadata required
  by fusion, features, and artifacts: `site_id`, IANA `site_timezone`,
  `profile_version`, `pipeline_version`, `cable_type`, `device_config`, and
  `rule_profile`. Defaults are stable (`unknown-site`, `UTC`, `profile-v1`,
  `pipeline-v1`, unknown cable/device, and an empty rule profile), so the original
  five-argument call remains valid. Task 11 is rules-only and records model version
  `rules-only`; later worker/model tasks may extend model inputs explicitly.
- Canonicalize and recursively freeze `rule_profile` at context construction, including
  nested lists and objects. Artifact evidence records the rule profile itself,
  `profile_version`, and its canonical SHA-256 so the rules-only decision can be
  reproduced.

- [ ] **Step 1: Add failing cross-block and idempotent-result tests**

```python
def test_pipeline_merges_event_across_block_boundary(tmp_path, boundary_h5, profile):
    result = analyze_file(boundary_h5, ZdHdf5Reader(), profile, select_backend("cpu"), tmp_path)
    assert len(result.events) == 1
    assert len(result.events[0].evidence) >= 2
    assert result.actual_backend == "cpu"

@pytest.mark.performance
def test_baseline_file_stays_under_memory_budget(baseline_h5, profile, tmp_path):
    result = measure_analysis(baseline_h5, profile, tmp_path)
    assert result.peak_rss_bytes < 4 * 1024**3
```

- [ ] **Step 2: Verify failure**

Run: `python -m pytest tests/test_pipeline.py -q`

Expected: FAIL importing `analyze_file`.

- [ ] **Step 3: Implement block iteration and one-file orchestration**

```python
@dataclass(frozen=True)
class FileAnalysis:
    source_path: str
    source_fingerprint: str
    actual_backend: str
    backend_fallback_reason: str | None
    metadata_warnings: tuple[str, ...]
    events: tuple[EventRecord, ...]
    audit_candidate_count: int
    timings_ms: dict[str, float]
    artifacts: tuple[ArtifactManifest, ...]
```

Use `reader.sample_count(path)` to include the final partial block without bypassing the
reader abstraction. Use 20-second logical blocks at a 15-second step. Prepend the required
LTA history only for STA/LTA detection; coherence, trajectory, and other detectors search
the logical block so a stronger history-only peak cannot suppress a canonical candidate.
Keep only candidates whose peak belongs to that block's canonical
`[logical_start, next_logical_start)` interval, with the last block owning through EOF;
this prevents both boundary blind zones and overlap duplicates.
Process one distinct detector frequency bank at a time, reuse it for detectors sharing
that band, then release it before the next band to bound memory. Run the enabled detectors,
translate to file-global indexes, apply deterministic same-detector safety deduplication
before audit/fusion (prefer unflagged, then higher-score evidence), fuse after all blocks,
extract features from event windows, classify, and render only fused review events. CUDA
arrays entering CPU-only trajectory, feature, or artifact paths use an explicit backend
transfer. For each rendered event, assemble one `EventArtifactData` from the
true raw samples and the matching filtered samples, preserving their shared sample/channel
coordinates plus UTC and site timezone metadata from `AnalysisContext`. Return audit
candidates separately from review events. `FileAnalysis` contains source fingerprint,
metadata warnings, events, audit candidate count, timings, actual backend, and artifact
manifests.

The source fingerprint is SHA-256 over a canonical JSON tuple of the resolved source path,
file size, and nanosecond modification time. Pass it into fusion and verify it after block
reads and before/following artifact rendering; fail the run if it changes. Do not hash the
complete DAS payload a second time. Validate the metadata sample rate as finite and
positive, and reject detector/energy bands outside its Nyquist range before processing.
Render every run into a private staging directory. Under the same promotion lock, perform
the final source revalidation and then atomically promote complete event directories.
Transfer/clear rollback ownership before releasing that lock after a successful commit;
no later failure may delete an already published directory without reacquiring the same
lock and proving ownership. On failure before commit, remove staging and only final
directories still exclusively owned by that failed run; never leave unregistered partial
artifacts.

- [ ] **Step 4: Run pipeline and opt-in performance tests**

Run: `python -m pytest tests/test_pipeline.py -q`

Expected: PASS.

Run: `python -m pytest tests/test_performance.py -m performance -q`

Expected: PASS on the configured baseline host for the approved logical baseline
`60 s x 1 kHz x 1000 channels`, including at least one bounded event artifact, with peak
RSS below 4 GiB. Use distinct approved frequency banks and execute real
energy/coherence/trajectory detector workspaces; a fixed detector may supply the single
bounded artifact event. Assert the approved CPU baseline wall-time ceiling of less than
60 seconds, while not asserting a universal GPU speedup.

- [ ] **Step 5: Commit**

```bash
git add backend/src/das_sentinel/pipeline.py tests/test_pipeline.py tests/test_performance.py
git commit -m "feat(pipeline): orchestrate chunked event analysis"
```

---

### Task 12: SQLite schema, migration, and repositories

**Files:**
- Create: `alembic.ini`
- Create: `backend/src/das_sentinel/persistence/__init__.py`
- Create: `backend/src/das_sentinel/persistence/db.py`
- Create: `backend/src/das_sentinel/persistence/models.py`
- Create: `backend/src/das_sentinel/persistence/repositories.py`
- Create: `backend/src/das_sentinel/persistence/migrations/env.py`
- Create: `backend/src/das_sentinel/persistence/migrations/versions/0001_initial.py`
- Create: `tests/test_persistence.py`

**Interfaces:**
- Produces repositories for admin, sites, profiles, source files, jobs, runs, events, reviews, models, and artifacts.

- [ ] **Step 1: Add failing migration and append-only review tests**

```python
def test_initial_migration_creates_all_tables(sqlite_url):
    upgrade_database(sqlite_url)
    assert set(inspect_tables(sqlite_url)) >= {
        "admins", "sites", "processing_profiles", "source_files", "jobs",
        "processing_runs", "events", "detection_evidence", "classifications",
        "reviews", "model_versions", "artifacts",
    }

def test_review_does_not_overwrite_machine_classification(repository, event_row):
    repository.add_review(event_row.id, label="earthquake", note="clear P/S")
    stored = repository.get_event(event_row.id)
    assert stored.machine_label == "unknown"
    assert stored.latest_review.label == "earthquake"
```

- [ ] **Step 2: Verify failure**

Run: `python -m pytest tests/test_persistence.py -q`

Expected: FAIL importing persistence modules.

- [ ] **Step 3: Implement schema and one-writer database setup**

Use SQLAlchemy 2 typed mappings, foreign keys, unique run keys, immutable profile JSON,
append-only reviews/classifications, UTC strings or timezone-aware conversion, and
transaction-scoped repository methods. On startup check SQLite version, set
`journal_mode=WAL`, `foreign_keys=ON`, `busy_timeout=5000`, and reject remote-looking
database URLs/paths. Model-version rows store the complete model-bundle SHA-256 separately
from the artifact path. Enforce append-only classifications/reviews and immutable profile
identity/payload in SQLite itself, not only through repository method omissions: database
triggers reject updates/deletes of classification/review rows and reject changes to a
profile's site, version, or payload while still allowing `is_current` to change. Removing a
parent must not silently cascade-delete append-only audit rows. Switching a site's current
profile clears and flushes the previous flag before setting the new flag, within the same
caller transaction, so SQLite's partial unique index cannot observe two current rows.
Canonical JSON rejects non-finite numbers. A programmatic migration URL takes precedence
over an ambient environment variable; the environment variable is only a CLI fallback when
the Alembic configuration URL is empty. Protect all writes with one async application lock
in the API layer.

- [ ] **Step 4: Run migrations twice and repository tests**

Run: `python -m pytest tests/test_persistence.py -q`

Expected: PASS; applying `upgrade head` twice is idempotent.

- [ ] **Step 5: Commit**

```bash
git add alembic.ini backend/src/das_sentinel/persistence tests/test_persistence.py
git commit -m "feat(db): add versioned SQLite persistence"
```

---

### Task 13: FastAPI shell, admin authentication, and site API

**Files:**
- Create: `backend/src/das_sentinel/api/__init__.py`
- Create: `backend/src/das_sentinel/api/app.py`
- Create: `backend/src/das_sentinel/api/auth.py`
- Create: `backend/src/das_sentinel/api/schemas.py`
- Create: `backend/src/das_sentinel/api/routes/auth.py`
- Create: `backend/src/das_sentinel/api/routes/sites.py`
- Create: `tests/test_auth_api.py`
- Create: `tests/test_sites_api.py`

**Interfaces:**
- Produces: `create_app(settings)`, `/api/v1/session`, `/api/v1/sites`.

- [ ] **Step 1: Add failing login, CSRF, and read-only path tests**

```python
def test_login_sets_http_only_session(client, seeded_admin):
    response = client.post("/api/v1/session", json={"password": "correct horse"})
    assert response.status_code == 204
    assert "HttpOnly" in response.headers["set-cookie"]

def test_mutation_requires_csrf(authenticated_client):
    response = authenticated_client.post("/api/v1/sites", json=valid_site())
    assert response.status_code == 403

def test_site_path_must_be_under_data_root(csrf_client):
    response = csrf_client.post("/api/v1/sites", json={**valid_site(), "data_path": "/etc"})
    assert response.status_code == 422
```

- [ ] **Step 2: Verify failure**

Run: `python -m pytest tests/test_auth_api.py tests/test_sites_api.py -q`

Expected: FAIL importing `create_app`.

- [ ] **Step 3: Implement single-admin sessions and site profile versions**

Hash with `argon2.PasswordHasher`; never return password hashes. Store admin ID and random
CSRF token in signed sessions. Clear any prior session on login, return the token in the
`X-CSRF-Token` response header, and let authenticated `GET /api/v1/session` recover the
same token. Session cookies are `HttpOnly`, `SameSite=Strict`, use the configured secure
flag, and never contain credentials or hashes. Except for the login endpoint, require an
authenticated, enabled admin and constant-time `X-CSRF-Token` validation for every
POST/PATCH/DELETE, including logout. Serialize write transactions through the one
application-scoped `AsyncWriteCoordinator`.

At application startup, validate the already-migrated local database with
`check_database_startup`; the web process does not silently run migrations or bypass the
SQLite safety gate. Resolve the configured data root and submitted site path with existing
filesystem entries, reject non-directories and symlink escapes, and ensure the result
remains under `Settings.data_root` without writing to it. Site edits create a new immutable
`ProcessingProfile` row and atomically mark it current.

- [ ] **Step 4: Run API tests**

Run: `python -m pytest tests/test_auth_api.py tests/test_sites_api.py -q`

Expected: PASS, including logout and disabled-site tests.

- [ ] **Step 5: Commit**

```bash
git add backend/src/das_sentinel/api tests/test_auth_api.py tests/test_sites_api.py
git commit -m "feat(api): add admin session and site management"
```

---

### Task 14: Persistent job queue and internal worker protocol

**Files:**
- Create: `backend/src/das_sentinel/services/jobs.py`
- Create: `backend/src/das_sentinel/api/routes/jobs.py`
- Create: `backend/src/das_sentinel/api/routes/internal.py`
- Create: `tests/test_jobs.py`
- Create: `tests/test_worker_protocol.py`

**Interfaces:**
- Produces: `JobService.enqueue/claim/heartbeat/complete/fail/recover_stale`, public job routes, internal worker routes.

- [ ] **Step 1: Add failing atomic-claim and stale-recovery tests**

```python
def test_only_one_worker_claims_job(job_service, queued_job):
    first = job_service.claim("worker-a")
    second = job_service.claim("worker-b")
    assert first.id == queued_job.id
    assert second is None

def test_stale_running_job_returns_to_queue(job_service, stale_job):
    recovered = job_service.recover_stale(now=stale_job.heartbeat_at + timedelta(minutes=3))
    assert recovered == [stale_job.id]
    assert job_service.get(stale_job.id).status == "queued"

def test_internal_route_rejects_wrong_token(client):
    assert client.post("/api/v1/internal/jobs/claim", headers={"X-Worker-Token": "bad"}).status_code == 401
```

- [ ] **Step 2: Verify failure**

Run: `python -m pytest tests/test_jobs.py tests/test_worker_protocol.py -q`

Expected: FAIL importing `JobService`.

- [ ] **Step 3: Implement short transactional state changes**

Use states `queued/running/succeeded/failed/cancelled`, an attempt counter, heartbeat,
progress numerator/denominator, error code/message, and retry eligibility. Claim and state
transitions execute under the API write lock. Internal submission validates worker token,
job ownership, pipeline/profile versions, and artifact paths below the configured root.

- [ ] **Step 4: Run job tests**

Run: `python -m pytest tests/test_jobs.py tests/test_worker_protocol.py -q`

Expected: PASS, including retry limit and cancellation tests.

- [ ] **Step 5: Commit**

```bash
git add backend/src/das_sentinel/services/jobs.py backend/src/das_sentinel/api/routes tests/test_jobs.py tests/test_worker_protocol.py
git commit -m "feat(jobs): add persistent queue and worker protocol"
```

---

### Task 15: Batch discovery and stable-file watcher

**Files:**
- Create: `backend/src/das_sentinel/services/discovery.py`
- Create: `tests/test_discovery.py`

**Interfaces:**
- Produces: `discover_batch(site, path)`, `StableFileTracker.observe`, `watch_site`.

- [ ] **Step 1: Add failing stability and fingerprint tests**

```python
def test_file_is_enqueued_only_after_two_stable_observations(tmp_path, tracker):
    path = tmp_path / "a.h5"
    write_zd_h5(path)
    assert tracker.observe(path, now=0) is None
    assert tracker.observe(path, now=5) is None
    stable = tracker.observe(path, now=10)
    assert stable.fingerprint == fingerprint(path)

def test_changed_file_gets_new_fingerprint(tmp_path):
    path = tmp_path / "a.h5"
    write_zd_h5(path)
    before = fingerprint(path)
    path.touch()
    assert fingerprint(path) != before
```

- [ ] **Step 2: Verify failure**

Run: `python -m pytest tests/test_discovery.py -q`

Expected: FAIL importing discovery functions.

- [ ] **Step 3: Implement polling without destructive file operations**

Use recursive or flat discovery according to the site profile, only include `.h5`, store
resolved paths, and calculate the approved path/size/mtime fingerprint. Require two stable
observations spanning 10 seconds and a successful lightweight `probe`. Persist discovered
files through the API repository and enqueue only missing `(fingerprint, profile_version,
pipeline_version)` run keys.

- [ ] **Step 4: Run discovery tests**

Run: `python -m pytest tests/test_discovery.py -q`

Expected: PASS, including partial HDF5 and symlink-escape rejection.

- [ ] **Step 5: Commit**

```bash
git add backend/src/das_sentinel/services/discovery.py tests/test_discovery.py
git commit -m "feat(discovery): add stable HDF5 batch and watch discovery"
```

---

### Task 16: Worker orchestration, CUDA fallback, and event APIs

**Files:**
- Create: `backend/src/das_sentinel/worker/__init__.py`
- Create: `backend/src/das_sentinel/worker/main.py`
- Create: `backend/src/das_sentinel/api/routes/events.py`
- Create: `backend/src/das_sentinel/api/routes/artifacts.py`
- Create: `tests/test_worker.py`
- Create: `tests/test_events_api.py`

**Interfaces:**
- Produces: `Worker.run_once`, event list/detail/review endpoints, safe artifact download.

- [ ] **Step 1: Add failing CPU fallback and append-only review tests**

```python
def test_worker_retries_cuda_oom_on_cpu(worker, cuda_oom_pipeline):
    result = worker.run_once()
    assert result.status == "succeeded"
    assert result.actual_backend == "cpu"
    assert result.backend_fallback_reason == "cuda_out_of_memory"

def test_event_review_preserves_machine_result(csrf_client, seeded_event):
    response = csrf_client.post(
        f"/api/v1/events/{seeded_event.id}/reviews",
        json={"label": "earthquake", "note": "visible P/S"},
    )
    assert response.status_code == 201
    event = csrf_client.get(f"/api/v1/events/{seeded_event.id}").json()
    assert event["machine_label"] == "unknown"
    assert event["latest_review"]["label"] == "earthquake"
```

- [ ] **Step 2: Verify failure**

Run: `python -m pytest tests/test_worker.py tests/test_events_api.py -q`

Expected: FAIL importing worker/event routes.

- [ ] **Step 3: Implement process-submit loop and safe event responses**

Worker claims one job, downloads immutable site/profile payload, resolves the reader,
chooses `Settings.worker_device` when it is not `auto` and otherwise uses the profile
device, calls `analyze_file`, and submits events/artifacts/timings.
Catch only recognized backend-unavailable/OOM exceptions for the one CPU retry; all other
exceptions become structured job failures. Event APIs filter by site/date/label/review
status and serve only registered artifacts whose resolved paths stay below Artifact Store.

- [ ] **Step 4: Run worker and event tests**

Run: `python -m pytest tests/test_worker.py tests/test_events_api.py -q`

Expected: PASS, including corrupt-file isolation and artifact traversal rejection.

- [ ] **Step 5: Commit**

```bash
git add backend/src/das_sentinel/worker backend/src/das_sentinel/api/routes tests/test_worker.py tests/test_events_api.py
git commit -m "feat(worker): process jobs and expose review events"
```

---

### Task 17: Training service and model API

**Files:**
- Create: `backend/src/das_sentinel/services/training.py`
- Create: `backend/src/das_sentinel/api/routes/models.py`
- Create: `tests/test_training_service.py`
- Create: `tests/test_models_api.py`

**Interfaces:**
- Produces: immutable training snapshots and `/api/v1/sites/{id}/models` train/publish/rollback routes.

- [ ] **Step 1: Add failing candidate/publish/rollback tests**

```python
def test_training_creates_candidate_not_active(training_service, reviewed_dataset):
    model = training_service.train(site_id=reviewed_dataset.site_id)
    assert model.status == "candidate"
    assert training_service.active_model(reviewed_dataset.site_id) is None

def test_publish_is_atomic_and_rollback_restores_previous(training_service, two_valid_models):
    first, second = two_valid_models
    training_service.publish(second.id)
    training_service.rollback(second.site_id)
    assert training_service.active_model(second.site_id).id == first.id

def test_model_below_gate_cannot_publish(csrf_client, weak_model):
    response = csrf_client.post(f"/api/v1/models/{weak_model.id}/publish")
    assert response.status_code == 409
```

- [ ] **Step 2: Verify failure**

Run: `python -m pytest tests/test_training_service.py tests/test_models_api.py -q`

Expected: FAIL importing `TrainingService`.

- [ ] **Step 3: Implement snapshots, reports, atomic activation, and rollback**

Snapshot reviewed event IDs, labels, ordered feature JSON, profile version, and SHA-256.
Training creates a model artifact, stores the SHA-256 returned by `save_model_bundle` in
the model-version database row, and creates the report without activation. Every model load
passes that trusted database digest to `load_model_bundle`. Publishing verifies the
four-class counts and metrics and refuses a candidate whose macro-F1 is below the active
model, deactivates the prior model and activates the candidate in one transaction. Rollback
activates the immediately preceding valid site model. Never delete model files during
rollback.

- [ ] **Step 4: Run training/model tests**

Run: `python -m pytest tests/test_training_service.py tests/test_models_api.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/src/das_sentinel/services/training.py backend/src/das_sentinel/api/routes/models.py tests/test_training_service.py tests/test_models_api.py
git commit -m "feat(models): add reviewed training publish and rollback"
```

---

### Task 18: CLI and system diagnostics

**Files:**
- Create: `backend/src/das_sentinel/cli.py`
- Create: `tests/test_cli.py`

**Interfaces:**
- Produces the approved `das-sentinel` commands and exit codes.

- [ ] **Step 1: Add failing CLI smoke tests**

```python
from typer.testing import CliRunner
from das_sentinel.cli import app

runner = CliRunner()

def test_doctor_reports_cpu_and_storage(tmp_path, monkeypatch):
    monkeypatch.setenv("DAS_APP_DATA_DIR", str(tmp_path))
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "CPU backend: available" in result.stdout
    assert "Application storage: writable" in result.stdout

def test_scan_calls_api(mock_api):
    result = runner.invoke(app, ["scan", "site-1", "/data/day"])
    assert result.exit_code == 0
    assert mock_api.last_request.path == "/api/v1/jobs"
```

- [ ] **Step 2: Verify failure**

Run: `python -m pytest tests/test_cli.py -q`

Expected: FAIL importing CLI app.

- [ ] **Step 3: Implement explicit command groups and JSON-friendly output**

Implement `admin init`, `site add/list/show`, `scan`, `watch`, `jobs
list/show/retry/cancel`, `events export`, `train`, `models list/publish/rollback`,
`benchmark`, and `doctor`. Network commands use the configured API URL/session token;
`doctor` validates SQLite version, local app volume, data-root read access, artifact
write access, CPU backend, optional CUDA, and worker token presence without printing
secrets.

- [ ] **Step 4: Run CLI tests**

Run: `python -m pytest tests/test_cli.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/src/das_sentinel/cli.py tests/test_cli.py
git commit -m "feat(cli): add administration analysis and diagnostics commands"
```

---

### Task 19: React shell, authentication, dashboard, sites, and jobs

**Files:**
- Modify: `backend/src/das_sentinel/api/app.py`
- Create: `frontend/package.json`
- Create: `frontend/tsconfig.json`
- Create: `frontend/vite.config.ts`
- Create: `frontend/index.html`
- Create: `frontend/src/main.tsx`
- Create: `frontend/src/App.tsx`
- Create: `frontend/src/api.ts`
- Create: `frontend/src/auth.tsx`
- Create: `frontend/src/styles.css`
- Create: `frontend/src/pages/LoginPage.tsx`
- Create: `frontend/src/pages/DashboardPage.tsx`
- Create: `frontend/src/pages/SitesPage.tsx`
- Create: `frontend/src/pages/JobsPage.tsx`
- Create: `frontend/src/pages/LoginPage.test.tsx`
- Create: `frontend/src/pages/SitesPage.test.tsx`
- Create: `frontend/src/pages/JobsPage.test.tsx`

**Interfaces:**
- Consumes the `/api/v1/session`, sites, and jobs JSON schemas.
- Produces authenticated SPA navigation and operational pages.

- [ ] **Step 1: Add failing component tests**

```tsx
it("logs in and redirects to the dashboard", async () => {
  server.use(http.post("/api/v1/session", () => new HttpResponse(null, {status: 204})));
  renderApp("/login");
  await userEvent.type(screen.getByLabelText("管理员密码"), "secret");
  await userEvent.click(screen.getByRole("button", {name: "登录"}));
  expect(await screen.findByRole("heading", {name: "运行总览"})).toBeVisible();
});

it("creates a site with explicit timezone and cable type", async () => {
  renderApp("/sites");
  await userEvent.click(await screen.findByRole("button", {name: "新建站点"}));
  await userEvent.type(screen.getByLabelText("名称"), "JJK-01");
  await userEvent.selectOptions(screen.getByLabelText("光缆类型"), "borehole");
  await userEvent.type(screen.getByLabelText("IANA 时区"), "Asia/Shanghai");
  await userEvent.click(screen.getByRole("button", {name: "保存"}));
  expect(await screen.findByText("JJK-01")).toBeVisible();
});
```

- [ ] **Step 2: Verify failure**

Run: `cd frontend && npm test -- --run`

Expected: FAIL because the React application files do not exist.

- [ ] **Step 3: Implement the dark operational shell and typed API client**

```json
{
  "name": "das-sentinel-web",
  "private": true,
  "version": "0.1.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc -b && vite build",
    "test": "vitest"
  },
  "dependencies": {
    "react": "^19.1.0",
    "react-dom": "^19.1.0",
    "react-router-dom": "^7.6.0"
  },
  "devDependencies": {
    "@testing-library/jest-dom": "^6.6.0",
    "@testing-library/react": "^16.3.0",
    "@testing-library/user-event": "^14.6.0",
    "@types/react": "^19.1.0",
    "@types/react-dom": "^19.1.0",
    "@vitejs/plugin-react": "^4.4.0",
    "jsdom": "^26.1.0",
    "msw": "^2.10.0",
    "typescript": "^5.8.0",
    "vite": "^7.0.0",
    "vitest": "^3.2.0"
  }
}
```

Use React Router, same-origin `fetch` with credentials, CSRF response storage, semantic
forms, Chinese labels, visible loading/error states, and keyboard-accessible navigation.
Implement the approved dashboard cards, site profile form, job progress table, retry and
cancel actions. Keep custom CSS focused; do not introduce a component framework. Mount
`frontend/dist` after `/api/v1` routes and return `index.html` for non-API client routes.

- [ ] **Step 4: Run frontend tests and production build**

Run: `cd frontend && npm test -- --run && npm run build`

Expected: PASS and `frontend/dist/index.html` exists.

- [ ] **Step 5: Commit**

```bash
git add frontend backend/src/das_sentinel/api/app.py
git commit -m "feat(web): add authenticated dashboard sites and jobs"
```

---

### Task 20: Three-column event review and model pages

**Files:**
- Create: `frontend/src/pages/EventsPage.tsx`
- Create: `frontend/src/pages/ModelsPage.tsx`
- Create: `frontend/src/components/EventQueue.tsx`
- Create: `frontend/src/components/ArtifactViewer.tsx`
- Create: `frontend/src/components/EvidencePanel.tsx`
- Create: `frontend/src/components/ReviewForm.tsx`
- Create: `frontend/src/components/ValidationReport.tsx`
- Create: `frontend/src/pages/EventsPage.test.tsx`
- Create: `frontend/src/pages/ModelsPage.test.tsx`

**Interfaces:**
- Consumes event/artifact/review/model API routes.
- Produces the approved review-first UI and manual model controls.

- [ ] **Step 1: Add failing review and publish tests**

```tsx
it("shows event queue artifacts and independent method evidence", async () => {
  renderApp("/events");
  await userEvent.click(await screen.findByText("2026-07-26 10:42:16"));
  expect(await screen.findByAltText("事件瀑布图")).toBeVisible();
  expect(screen.getByRole("tab", {name: "PSD"})).toBeVisible();
  expect(screen.getByText("STA/LTA 多道符合")).toBeVisible();
  expect(screen.getByText("规则判断")).toBeVisible();
});

it("requires explicit confirmation before model publication", async () => {
  renderApp("/models");
  await userEvent.click(await screen.findByRole("button", {name: "发布候选模型"}));
  expect(screen.getByRole("dialog", {name: "确认发布模型"})).toBeVisible();
});
```

- [ ] **Step 2: Verify failure**

Run: `cd frontend && npm test -- --run src/pages/EventsPage.test.tsx src/pages/ModelsPage.test.tsx`

Expected: FAIL because pages/components are missing.

- [ ] **Step 3: Implement queue filters, artifact tabs, evidence, review, and models**

Use a 30% event queue and 70% detail area; the detail area places waterfall above PSD/STFT
and method evidence beside the plots at wide sizes, stacking on narrow screens. Preserve
machine label while displaying latest human review. Support label/note submission, similar
event links, training trigger, validation matrix, publish confirmation, and rollback
confirmation. Images open full-screen and have download links.

- [ ] **Step 4: Run all frontend tests**

Run: `cd frontend && npm test -- --run && npm run build`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add frontend/src
git commit -m "feat(web): add three-column review and model lifecycle"
```

---

### Task 21: CPU/GPU containers and Compose deployment

**Files:**
- Create: `docker/Dockerfile.cpu`
- Create: `docker/Dockerfile.gpu`
- Create: `docker/entrypoint-api.sh`
- Create: `compose.yaml`
- Create: `compose.cpu.yaml`
- Create: `compose.gpu.yaml`
- Create: `.env.example`
- Create: `tests/test_compose.py`

**Interfaces:**
- Produces independent CPU and GPU deployment commands with the same API/app-data layout.

- [ ] **Step 1: Add failing deployment configuration assertions**

```python
def test_raw_data_mounts_are_read_only(compose_cpu):
    api_raw = next(v for v in compose_cpu["services"]["api"]["volumes"] if v["target"] == "/data")
    worker_raw = next(v for v in compose_cpu["services"]["worker"]["volumes"] if v["target"] == "/data")
    assert api_raw["read_only"] is True
    assert worker_raw["read_only"] is True

def test_gpu_worker_reserves_nvidia_device(compose_gpu):
    devices = compose_gpu["services"]["worker"]["deploy"]["resources"]["reservations"]["devices"]
    assert devices == [{"driver": "nvidia", "count": 1, "capabilities": ["gpu"]}]
```

- [ ] **Step 2: Verify failure**

Run: `python -m pytest tests/test_compose.py -q`

Expected: FAIL because Compose files do not exist.

- [ ] **Step 3: Implement multi-stage frontend/backend images and health checks**

```yaml
# compose.yaml
services:
  api:
    build:
      context: .
      dockerfile: docker/Dockerfile.cpu
    command: ["api"]
    env_file: [.env]
    ports: ["8000:8000"]
    volumes:
      - type: bind
        source: ${DAS_DATA_ROOT}
        target: /data
        read_only: true
      - type: volume
        source: app-data
        target: /app-data
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz')"]
      interval: 15s
      timeout: 5s
      retries: 5
volumes:
  app-data: {}
```

```yaml
# compose.cpu.yaml
services:
  worker:
    build:
      context: .
      dockerfile: docker/Dockerfile.cpu
    command: ["worker"]
    env_file: [.env]
    depends_on:
      api:
        condition: service_healthy
    volumes:
      - type: bind
        source: ${DAS_DATA_ROOT}
        target: /data
        read_only: true
      - type: volume
        source: app-data
        target: /app-data
volumes:
  app-data: {}
```

```yaml
# compose.gpu.yaml
services:
  worker:
    build:
      context: .
      dockerfile: docker/Dockerfile.gpu
    command: ["worker"]
    env_file: [.env]
    environment:
      DAS_WORKER_DEVICE: cuda
    depends_on:
      api:
        condition: service_healthy
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    volumes:
      - type: bind
        source: ${DAS_DATA_ROOT}
        target: /data
        read_only: true
      - type: volume
        source: app-data
        target: /app-data
volumes:
  app-data: {}
```

Build React with Node 24, copy `frontend/dist` into the Python 3.12 API image, install the
Python package from `pyproject.toml`, run Alembic before Uvicorn, and run as a
non-root UID. CPU override starts a CPU worker. GPU override builds from a compatible
NVIDIA CUDA runtime, installs `cupy-cuda12x`, reserves one GPU, and starts with
`DAS_DEVICE=cuda`. Mount `/data:ro` and `/app-data:rw`; keep SQLite under `/app-data`,
not `/data`.

- [ ] **Step 4: Validate configs and build CPU image**

Run: `docker compose -f compose.yaml -f compose.cpu.yaml config`

Expected: valid configuration with one API and one CPU worker.

Run: `docker compose -f compose.yaml -f compose.cpu.yaml build`

Expected: both CPU services build. On a CUDA host also run
`docker compose -f compose.yaml -f compose.gpu.yaml config`.

Run:

```bash
docker compose -f compose.yaml -f compose.cpu.yaml run --rm --no-deps worker \
  python -c "import importlib.util; assert importlib.util.find_spec('cupy') is None"
```

Expected: PASS, proving from the built CPU container's runtime environment that CuPy
is not installed.

- [ ] **Step 5: Commit**

```bash
git add docker compose.yaml compose.cpu.yaml compose.gpu.yaml .env.example tests/test_compose.py
git commit -m "feat(deploy): add NAS CPU and NVIDIA GPU containers"
```

---

### Task 22: End-to-end acceptance, performance evidence, and operator docs

**Files:**
- Create: `e2e/playwright.config.ts`
- Create: `e2e/review-flow.spec.ts`
- Create: `tests/test_fault_injection.py`
- Create: `scripts/benchmark_baseline.py`
- Create: `README.md`
- Create: `docs/operations.md`
- Create: `docs/site-calibration.md`
- Create: `docs/acceptance-report.md`

**Interfaces:**
- Produces reproducible install, calibration, recovery, benchmark, and acceptance evidence.

- [ ] **Step 1: Add failing browser and fault-injection tests**

```ts
test("admin scans, reviews, trains, and inspects the validation gate", async ({page}) => {
  await page.goto("/login");
  await page.getByLabel("管理员密码").fill("test-admin-password");
  await page.getByRole("button", {name: "登录"}).click();
  await page.getByRole("link", {name: "事件复核"}).click();
  await page.getByText("synthetic-earthquake").click();
  await expect(page.getByAltText("事件瀑布图")).toBeVisible();
  await page.getByLabel("人工标签").selectOption("earthquake");
  await page.getByRole("button", {name: "保存复核"}).click();
  await expect(page.getByText("已保存")).toBeVisible();
});
```

```python
def test_corrupt_file_fails_without_blocking_next_file(running_stack, corrupt_h5, valid_h5):
    job = running_stack.scan([corrupt_h5, valid_h5])
    running_stack.wait(job)
    assert running_stack.file_status(corrupt_h5) == "failed"
    assert running_stack.file_status(valid_h5) == "succeeded"
```

- [ ] **Step 2: Run tests and verify missing acceptance fixtures fail**

Run: `python -m pytest tests/test_fault_injection.py -q`

Run: `cd e2e && npx playwright test`

Expected: FAIL until the test stack and seeded fixtures are wired.

- [ ] **Step 3: Wire the test stack, benchmark, and exact documentation**

Add a Compose test profile that seeds one admin, one site, synthetic HDF5 files for
earthquake/blast/traffic/common-mode/single-channel/boundary cases, and deterministic
reviewed training rows. `benchmark_baseline.py` generates or reads a
`1000 channels × 1000 Hz × 60 s` file, records CPU/GPU wall time, peak RSS/VRAM, backend,
versions, and parity. README includes CPU/GPU startup commands. Operations covers backup
of `/app-data`, restart recovery, failed-file retry, health checks, and no raw-data
deletion. Calibration explains why city/submarine accuracy is unclaimed without local
samples and how to tune a new site.

- [ ] **Step 4: Run the full acceptance sequence**

Run:

```bash
python -m pytest -q
ruff check backend tests
mypy backend/src/das_sentinel
cd frontend && npm test -- --run && npm run build
cd ../e2e && npx playwright test
cd .. && docker compose -f compose.yaml -f compose.cpu.yaml up -d --build
python scripts/benchmark_baseline.py --device cpu --out docs/acceptance-report.md
docker compose -f compose.yaml -f compose.cpu.yaml down
```

Expected:

- all Python, frontend, and browser tests pass;
- CPU stack becomes healthy;
- benchmark reports peak RSS below 4 GB and mean baseline processing below 60 seconds on
  the designated 4-CPU acceptance host;
- acceptance report identifies real hardware and never invents city/submarine accuracy.

- [ ] **Step 5: Perform the real read-only HDF5 smoke test and commit**

Run `das-sentinel benchmark` against one explicitly selected existing ZD HDF5 file under
`Z:\jjk*` with output directed only to the new project's Artifact Store. Verify source
size, modification time, and SHA-256 are unchanged before and after.

```bash
git add e2e tests/test_fault_injection.py scripts README.md docs
git commit -m "test: complete DAS Sentinel acceptance workflow"
```

---

## Plan Self-Review

### Spec coverage

- Input plugin, HDF5, time and geometry: Tasks 1–2.
- Chunking, preprocessing, CPU/CUDA: Tasks 3 and 11.
- Four detection families and balanced fusion: Tasks 4–7.
- Features, three classification methods, abstention and model gates: Tasks 8–9 and 17.
- Waterfall, waveform, PSD, STFT and evidence: Task 10.
- Idempotent jobs, stable files, restart and failure isolation: Tasks 12, 14–16.
- Single-admin security and multi-site profiles: Task 13.
- Web review-first layout and model controls: Tasks 19–20.
- CLI: Task 18.
- CPU/GPU independent deployment and read-only raw mounts: Task 21.
- Synthetic, real-data, fault, end-to-end and performance acceptance: Task 22.
- Explicit non-goals remain absent: no remote worker cluster, GIS, source inversion,
  multi-user roles, acquisition protocol, or unsupported reader implementation.

### Type consistency

- `DasBlock`, `Candidate`, `EventRecord`, `ProcessingProfile`, and `ComputeBackend` originate
  in Task 1 and retain the same names in every later task.
- Detectors always return `list[Candidate]`; fusion returns `list[EventRecord]`.
- `FileAnalysis` is the only worker submission payload for a processed source file.
- `unknown` appears only in rule/model fusion and persisted machine output, never in
  supervised training labels.
- Worker never imports persistence repositories; it communicates only through internal API.

### Placeholder scan

The plan contains no deferred code markers, unspecified error-handling steps, or undefined
public functions. Each task names its files, inputs, outputs, focused failure, verification
command, and commit.

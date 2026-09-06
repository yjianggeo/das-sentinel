# Task 4 CLI automatic scan report

## TDD evidence

### RED

Command, run after adding the Task 4 CLI tests and before changing production
code:

```powershell
& 'D:\das-sentinel-runtime\.venv\Scripts\python.exe' -m pytest tests/test_cli.py -k 'scan or watch' -q
```

Observed output before the implementation:

```text
.FFFF
```

The old CLI slept for 10 seconds for each scan invocation, so the interactive
30-second command window ended before pytest emitted its summary. The failures
were expected: the old commands accepted a missing PATH, owned the settle
delay locally, and issued a second scan POST.

### GREEN

Focused command:

```powershell
& 'D:\das-sentinel-runtime\.venv\Scripts\python.exe' -m pytest tests/test_cli.py -k 'scan or watch' -q
```

Output:

```text
..........                                                               [100%]
10 passed, 8 deselected in 0.74s
```

Complete CLI command:

```powershell
& 'D:\das-sentinel-runtime\.venv\Scripts\python.exe' -m pytest tests/test_cli.py -q
```

Output:

```text
..................                                                       [100%]
18 passed in 4.53s
```

Additional focused static check:

```powershell
& 'D:\das-sentinel-runtime\.venv\Scripts\python.exe' -m ruff check backend/src/das_sentinel/cli.py tests/test_cli.py
```

Output:

```text
All checks passed!
```

## Changed files

- `backend/src/das_sentinel/cli.py`: requires a concrete PATH, normalizes it
  to a root-relative POSIX path, performs one POST, polls active server scan
  phases, reports Chinese scan status, and exits 1 on server failure.
- `tests/test_cli.py`: queue-capable `FakeApi` plus regression coverage for
  one POST, polling, required/rejected/absolute paths, server failure, and
  `watch --once`.
- `docs/site-calibration.md`: explains the required non-root PATH and
  server-owned 10-second stability recheck.

## Self-review

- Checked every logical scan path submits exactly one POST; the focused tests
  count recorded `/scan` requests for `scan` and `watch --once`.
- Checked shared normalization rejects root and parent traversal before POST,
  permits in-root absolute Windows paths, and emits POSIX payload paths.
- Checked `watch --once` waits through the terminal snapshot before returning;
  recurring `watch` applies its poll delay only afterward.
- No API, service, dependency, or runtime files were changed.

## Follow-up concerns

- The CLI intentionally trusts the reviewed API's six scan phases and status
  fields. A future API phase would need an explicit CLI message/terminal rule.

## Review fix round 1/5: Windows rooted paths

### Root cause

On Windows, both `Path('/outside')` and `Path('\\outside')` have a lexical
root anchor but report `is_absolute() == False`. The previous normalization
therefore treated them as site-relative values and issued the scan POST.

### RED

The path rejection table was extended with `/outside` and `\\outside`, while
retaining the assertion that no recorded request is a POST. Before the fix:

```powershell
& 'D:\das-sentinel-runtime\.venv\Scripts\python.exe' -m pytest tests/test_cli.py -k rooted -q --tb=short
```

```text
...FF                                                                    [100%]
FAILED ...[/outside] - assert 1 == 2
FAILED ...[\\outside] - assert 1 == 2
2 failed, 3 passed, 15 deselected in 0.83s
```

### GREEN and verification

The CLI now checks both `PurePosixPath(value).anchor` and
`PureWindowsPath(value).anchor` before accepting a non-absolute value as
relative. Native absolute paths still use the existing in-root `relative_to`
normalization.

```powershell
& 'D:\das-sentinel-runtime\.venv\Scripts\python.exe' -m pytest tests/test_cli.py -k rooted -q
```

```text
.....                                                                    [100%]
5 passed, 15 deselected in 0.65s
```

```powershell
& 'D:\das-sentinel-runtime\.venv\Scripts\python.exe' -m pytest tests/test_cli.py -q
& 'D:\das-sentinel-runtime\.venv\Scripts\python.exe' -m ruff check backend/src/das_sentinel/cli.py tests/test_cli.py
```

```text
....................                                                     [100%]
20 passed in 4.38s
All checks passed!
```

### Self-review

- The added check only rejects anchored values that the host platform cannot
  interpret as absolute; valid `C:/data/...` site paths remain supported.
- The shared helper protects both `scan` and `watch` before their first POST.
- No API, service, dependency, or runtime changes were made.

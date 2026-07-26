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

    def read_block(
        self,
        path: Path,
        sample_start: int,
        sample_stop: int,
        channel_slice: slice = slice(None),
    ) -> DasBlock: ...


class ReaderRegistry:
    def __init__(self, readers: list[DasReader]) -> None:
        self._readers = readers

    def resolve(self, path: Path) -> DasReader:
        for reader in self._readers:
            if reader.probe(path).supported:
                return reader
        raise ValueError(f"No DAS reader supports {path}")

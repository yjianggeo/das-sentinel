from datetime import UTC, datetime

import numpy as np
import pytest
from das_sentinel.domain import DasBlock, DasMetadata


def test_das_block_requires_time_channel_float32():
    metadata = DasMetadata(
        source_path="/data/a.h5",
        start_time_utc=datetime.now(UTC),
        sample_rate_hz=1000.0,
        channel_distance_m=np.arange(3, dtype=np.float32),
        unit="strain_rate",
    )

    with pytest.raises(ValueError, match="float32"):
        DasBlock(
            data=np.zeros((10, 3), dtype=np.float64),
            metadata=metadata,
            sample_offset=0,
        )

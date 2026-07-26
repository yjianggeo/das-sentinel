import pytest
from das_sentinel.config import ProcessingProfile
from pydantic import ValidationError


def test_profile_has_approved_block_defaults():
    profile = ProcessingProfile()

    assert profile.block_seconds == 20.0
    assert profile.overlap_seconds == 5.0
    assert profile.device == "auto"


def test_profile_rejects_overlap_not_smaller_than_block():
    with pytest.raises(ValidationError):
        ProcessingProfile(block_seconds=5.0, overlap_seconds=5.0)

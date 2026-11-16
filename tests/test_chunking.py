import pytest

from chunking import default_chunk_config, validate_chunk_config


def test_default_chunk_config_matches_project_defaults():
    config = default_chunk_config()
    assert config.parent_chunk_size == 1000
    assert config.parent_chunk_overlap == 200
    assert config.child_chunk_size == 200
    assert config.child_chunk_overlap == 50


def test_chunk_config_accepts_non_overlapping_child_chunks():
    config = validate_chunk_config(
        parent_chunk_size=1000,
        parent_chunk_overlap=200,
        child_chunk_size=200,
        child_chunk_overlap=0,
    )
    assert config.child_chunk_overlap == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"child_chunk_size": 10},
        {"child_chunk_size": 1200, "parent_chunk_size": 1000},
        {"child_chunk_overlap": 200, "child_chunk_size": 200},
        {"parent_chunk_overlap": 1000, "parent_chunk_size": 1000},
    ],
)
def test_chunk_config_rejects_invalid_bounds(kwargs):
    with pytest.raises(ValueError):
        validate_chunk_config(**kwargs)

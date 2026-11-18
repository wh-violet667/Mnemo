import pytest

from chunking import (
    default_chunk_config,
    list_named_chunk_configs,
    named_chunk_config,
    validate_chunk_config,
)


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


def test_default_named_profile_matches_default_config():
    assert named_chunk_config() == default_chunk_config()


def test_named_profiles_cover_fine_and_coarse_strategies():
    fine = named_chunk_config("fine-overlap")
    coarse = named_chunk_config("coarse-no-overlap")

    assert fine.child_chunk_size == 200
    assert fine.child_chunk_overlap == 50
    assert coarse.child_chunk_size == 400
    assert coarse.child_chunk_overlap == 0
    assert len(list_named_chunk_configs()) == 2


def test_unknown_named_profile_is_rejected():
    with pytest.raises(ValueError, match="未知分块策略"):
        named_chunk_config("missing-profile")


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

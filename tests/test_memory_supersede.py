from memory_manager import select_superseded_candidates


def test_selects_only_candidates_above_threshold():
    mapping = select_superseded_candidates(
        [(0, "old-a", 0.91), (1, "old-b", 0.89)],
        threshold=0.9,
    )

    assert mapping == {0: {"memory_id": "old-a", "score": 0.91}}


def test_same_old_memory_is_claimed_by_highest_score_only():
    mapping = select_superseded_candidates(
        [
            (0, "old-a", 0.91),
            (1, "old-a", 0.99),
            (2, "old-b", 0.95),
        ],
        threshold=0.9,
    )

    assert mapping[1]["memory_id"] == "old-a"
    assert mapping[2]["memory_id"] == "old-b"
    assert 0 not in mapping

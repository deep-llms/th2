"""Focused unit tests for the Hashed-v2 quota artifact generator."""

import numpy as np
import pytest

from scripts.make_token_importance_quota import (
    _stable_descending_order,
    coverage_quota_order,
    importance_from_order,
)


def test_floor_quota_preserves_v1_prefix_and_places_zero_mass_last():
    shares = np.array(
        [
            [0.4, 0.3, 0.2, 0.1, 0.0, 0.0],
            [0.1, 0.1, 0.6, 0.2, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    v1_importance = np.array([10.0, 4.0, 8.0, 2.0, 0.0, 0.0])
    ordering = coverage_quota_order(shares, v1_importance, v1_head_size=1)
    assert ordering.tolist() == [0, 2, 1, 3, 4, 5]
    importance = importance_from_order(ordering)
    assert importance.dtype == np.float64
    assert np.array_equal(_stable_descending_order(importance), ordering)


def test_floor_quota_is_deterministic_and_rejects_unnormalized_shares():
    shares = np.array(
        [
            [0.5, 0.25, 0.25, 0.0],
            [0.25, 0.5, 0.25, 0.0],
        ],
        dtype=np.float64,
    )
    importance = np.array([4.0, 3.0, 2.0, 1.0])
    first = coverage_quota_order(shares, importance, v1_head_size=1)
    second = coverage_quota_order(shares, importance, v1_head_size=1)
    assert np.array_equal(first, second)
    with pytest.raises(ValueError, match="sum to one"):
        coverage_quota_order(shares * 2.0, importance, v1_head_size=1)

"""
tests/test_match_state.py — Milestone 2: Match State Tests
===========================================================

Tests for src/match_state.py:
- state_from_delivery_row() correctness against manually computed values.
- state_to_observation() shape, dtype, and normalization bounds.
- Edge cases: last ball, target passed, innings 1 (no target).

Run with:
    pytest tests/test_match_state.py -v
"""

import math
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from match_state import MatchState, state_from_delivery_row, state_to_observation

DATA = Path(__file__).resolve().parents[1] / "data" / "processed"


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def deliveries_clean():
    """Load the full deliveries_clean.csv once for the session."""
    return pd.read_csv(DATA / "deliveries_clean.csv", low_memory=False)


@pytest.fixture(scope="session")
def first_innings2_row(deliveries_clean):
    """First delivery of any second-innings (match 1082591, innings 2, over 0)."""
    row = deliveries_clean[deliveries_clean["innings"] == 2].iloc[0]
    return row


@pytest.fixture(scope="session")
def first_innings1_row(deliveries_clean):
    """First delivery of any first innings (no target)."""
    return deliveries_clean[deliveries_clean["innings"] == 1].iloc[0]


# ─────────────────────────────────────────────────────────────────────────────
# Known-value regression tests
# ─────────────────────────────────────────────────────────────────────────────

class TestKnownValues:
    """
    Regression tests against manually computed state values.

    We use match 1082591 innings 2 as the reference case.
    From the audit: innings 2 target = 208, score at ball 1 = 0.
    """

    def test_balls_remaining_at_start_of_innings(self, first_innings2_row):
        """At the very first ball of innings 2, legal_balls_before = 0 → 120 remaining."""
        state = state_from_delivery_row(first_innings2_row)
        assert state.balls_remaining == 120, (
            f"Expected 120 balls remaining at start of innings, got {state.balls_remaining}"
        )

    def test_current_score_at_start(self, first_innings2_row):
        """Score at start of innings must be 0."""
        state = state_from_delivery_row(first_innings2_row)
        assert state.current_score == 0

    def test_wickets_at_start(self, first_innings2_row):
        """Wickets at start of innings must be 0."""
        state = state_from_delivery_row(first_innings2_row)
        assert state.wickets_lost == 0

    def test_target_is_correct(self, first_innings2_row):
        """Target for match 1082591 innings 2 must be 208."""
        state = state_from_delivery_row(first_innings2_row)
        assert state.target == 208.0, (
            f"Expected target=208.0, got {state.target}"
        )

    def test_runs_required_at_start(self, first_innings2_row):
        """runs_required at start of innings = target - 0 = 208."""
        state = state_from_delivery_row(first_innings2_row)
        assert state.runs_required == 208.0

    def test_required_run_rate_at_start(self, first_innings2_row):
        """
        RRR at start = (208 * 6) / 120 = 10.4.
        This is the runs-per-over rate required to win from ball 1.
        """
        state = state_from_delivery_row(first_innings2_row)
        expected_rrr = (208.0 * 6.0) / 120.0
        assert abs(state.required_run_rate - expected_rrr) < 1e-6, (
            f"Expected RRR={expected_rrr:.4f}, got {state.required_run_rate:.4f}"
        )


class TestManualRowDict:
    """Tests using synthetic dict rows so we can control every value precisely."""

    def _make_row(self, legal_balls_before, runs_before, wickets_before,
                  target=None, runs_required_before=None):
        """Build a minimal synthetic delivery row as a dict."""
        return {
            "legal_balls_before": legal_balls_before,
            "runs_before": runs_before,
            "wickets_before": wickets_before,
            "target": target if target is not None else float("nan"),
            "runs_required_before": (
                runs_required_before if runs_required_before is not None
                else float("nan")
            ),
        }

    def test_balls_remaining_formula(self):
        """balls_remaining = 120 - legal_balls_before."""
        row = self._make_row(60, 80, 2, target=160, runs_required_before=80)
        state = state_from_delivery_row(row)
        assert state.balls_remaining == 60  # 120 - 60

    def test_rrr_formula(self):
        """RRR = (runs_required * 6) / balls_remaining."""
        # 60 runs needed, 30 balls left → RRR = (60 * 6) / 30 = 12.0
        row = self._make_row(90, 100, 3, target=160, runs_required_before=60)
        state = state_from_delivery_row(row)
        assert abs(state.required_run_rate - 12.0) < 1e-6

    def test_rrr_target_passed(self):
        """When runs_required <= 0, RRR should be 0.0 (batting team has won)."""
        row = self._make_row(60, 165, 2, target=160, runs_required_before=-5)
        state = state_from_delivery_row(row)
        assert state.required_run_rate == 0.0

    def test_rrr_sentinel_at_last_ball(self):
        """When balls_remaining == 0, RRR = sentinel 99.0."""
        row = self._make_row(120, 150, 5, target=160, runs_required_before=10)
        state = state_from_delivery_row(row)
        assert state.required_run_rate == MatchState._RRR_SENTINEL

    def test_rrr_nan_for_innings1(self):
        """For innings 1 (no target), all chase features must be NaN."""
        row = self._make_row(30, 50, 1, target=None, runs_required_before=None)
        state = state_from_delivery_row(row)
        assert math.isnan(state.target)
        assert math.isnan(state.runs_required)
        assert math.isnan(state.required_run_rate)

    def test_raises_on_negative_balls_remaining(self):
        """
        legal_balls_before > 120 should raise ValueError.
        This guards against super-over rows being accidentally included.
        """
        row = self._make_row(121, 100, 0, target=150, runs_required_before=50)
        with pytest.raises(ValueError, match="balls_remaining.*negative"):
            state_from_delivery_row(row)


# ─────────────────────────────────────────────────────────────────────────────
# state_to_observation() tests
# ─────────────────────────────────────────────────────────────────────────────

class TestStateToObservation:
    """Tests for the normalized numpy observation vector."""

    def _make_state(self, current_score=0, wickets_lost=0, balls_remaining=120,
                    target=200.0, runs_required=200.0, required_run_rate=10.0):
        return MatchState(
            current_score=current_score,
            wickets_lost=wickets_lost,
            balls_remaining=balls_remaining,
            target=target,
            runs_required=runs_required,
            required_run_rate=required_run_rate,
        )

    def test_output_shape(self):
        """Observation must be a 1D array of length 6."""
        obs = state_to_observation(self._make_state())
        assert obs.shape == (6,), f"Expected shape (6,), got {obs.shape}"

    def test_output_dtype(self):
        """Observation must be float32."""
        obs = state_to_observation(self._make_state())
        assert obs.dtype == np.float32, f"Expected float32, got {obs.dtype}"

    def test_all_finite(self):
        """All elements must be finite (no NaN, no Inf)."""
        obs = state_to_observation(self._make_state())
        assert np.all(np.isfinite(obs)), (
            f"Observation contains non-finite values: {obs}"
        )

    def test_all_in_zero_one(self):
        """All elements must be in [0, 1] after normalization."""
        obs = state_to_observation(self._make_state())
        assert np.all(obs >= 0.0) and np.all(obs <= 1.0), (
            f"Observation out of [0, 1]: {obs}"
        )

    def test_clipping_extreme_values(self):
        """Values above the normalization max should be clipped to 1.0."""
        state = self._make_state(current_score=9999, target=9999, runs_required=9999)
        obs = state_to_observation(state)
        assert np.all(obs <= 1.0), (
            f"Clipping failed: observation exceeds 1.0: {obs}"
        )

    def test_nan_treated_as_zero(self):
        """NaN features (innings 1 chase features) must become 0.0, not NaN."""
        state = MatchState(
            current_score=50, wickets_lost=2, balls_remaining=60,
            target=float("nan"), runs_required=float("nan"),
            required_run_rate=float("nan"),
        )
        obs = state_to_observation(state)
        assert np.all(np.isfinite(obs)), (
            f"NaN in observation vector: {obs}"
        )
        # Chase-specific features (indices 3, 4, 5) must be 0.0.
        assert obs[3] == 0.0  # target
        assert obs[4] == 0.0  # runs_required
        assert obs[5] == 0.0  # required_run_rate

    def test_zero_state_gives_zero_observation(self):
        """All-zero state (start of innings with no chase) maps to [0, 0, 1, 0, 0, 0]."""
        # balls_remaining = 120, normalized = 120/120 = 1.0
        state = MatchState(
            current_score=0, wickets_lost=0, balls_remaining=120,
            target=float("nan"), runs_required=float("nan"),
            required_run_rate=float("nan"),
        )
        obs = state_to_observation(state)
        assert obs[0] == 0.0   # score
        assert obs[1] == 0.0   # wickets
        assert abs(obs[2] - 1.0) < 1e-6  # balls_remaining (full innings)
        assert obs[3] == 0.0   # target (NaN → 0)
        assert obs[4] == 0.0   # runs_required (NaN → 0)
        assert obs[5] == 0.0   # rrr (NaN → 0)

    def test_feature_ordering(self):
        """
        Feature index ordering must be stable (changing it would break
        saved DQN checkpoints). Verify each index maps to the right feature.
        Index 0: current_score, 1: wickets_lost, 2: balls_remaining,
        3: target, 4: runs_required, 5: required_run_rate.
        """
        state = MatchState(
            current_score=125,  # 125/250 = 0.5
            wickets_lost=5,     # 5/10 = 0.5
            balls_remaining=60, # 60/120 = 0.5
            target=150.0,       # 150/300 = 0.5
            runs_required=150.0, # 150/300 = 0.5
            required_run_rate=18.0,  # 18/36 = 0.5
        )
        obs = state_to_observation(state)
        for i, expected in enumerate([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]):
            assert abs(obs[i] - expected) < 1e-5, (
                f"Feature index {i}: expected {expected}, got {obs[i]}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Integration test: real data round-trip
# ─────────────────────────────────────────────────────────────────────────────

class TestRealDataRoundTrip:
    """Verify state construction + observation on a sample of real rows."""

    SAMPLE_N = 500  # rows to sample

    def test_all_observations_finite_on_real_data(self, deliveries_clean):
        """
        For a sample of real delivery rows, every observation vector must be
        finite (no NaN, no Inf). This catches any missed edge cases in the
        actual data that our synthetic tests might not cover.
        """
        inn2_rows = deliveries_clean[deliveries_clean["innings"] == 2].sample(
            n=min(self.SAMPLE_N, len(deliveries_clean)), random_state=42
        )
        failures = []
        for idx, row in inn2_rows.iterrows():
            try:
                state = state_from_delivery_row(row)
                obs = state_to_observation(state)
                if not np.all(np.isfinite(obs)):
                    failures.append((idx, obs))
            except (ValueError, KeyError) as e:
                failures.append((idx, str(e)))

        assert not failures, (
            f"{len(failures)} rows produced non-finite observations. "
            f"First failure: {failures[0]}"
        )

    def test_all_observations_in_range_on_real_data(self, deliveries_clean):
        """All normalized values must be in [0, 1]."""
        inn2_rows = deliveries_clean[deliveries_clean["innings"] == 2].sample(
            n=min(self.SAMPLE_N, len(deliveries_clean)), random_state=42
        )
        out_of_range = 0
        for idx, row in inn2_rows.iterrows():
            state = state_from_delivery_row(row)
            obs = state_to_observation(state)
            if np.any(obs < 0.0) or np.any(obs > 1.0):
                out_of_range += 1

        assert out_of_range == 0, (
            f"{out_of_range} rows produced observations outside [0, 1]"
        )

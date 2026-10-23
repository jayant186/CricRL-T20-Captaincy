import math
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from chase_features import (
    build_chase_features,
    PROCESSED_DIR,
    OUTPUT_FULL_PATH,
    OUTPUT_INNINGS2_PATH,
    RRR_SENTINEL,
    TOTAL_LEGAL_BALLS,
)

DATA = Path(__file__).resolve().parents[1] / "data" / "processed"

# Session-scoped fixtures: build once, share across all tests.
@pytest.fixture(scope="session")
def chase_output():
    """Run the full build pipeline once per session."""
    full_df, innings2_df = build_chase_features(verbose=False)
    return full_df, innings2_df


@pytest.fixture(scope="session")
def full_df(chase_output):
    return chase_output[0]


@pytest.fixture(scope="session")
def innings2_df(chase_output):
    return chase_output[1]

# Output file existence
class TestOutputFiles:
    """Both output CSV files must be written by the build pipeline."""

    def test_deliveries_chase_exists(self, chase_output):
        assert OUTPUT_FULL_PATH.exists(), (
            f"Expected output file not found: {OUTPUT_FULL_PATH}"
        )

    def test_innings2_chase_exists(self, chase_output):
        assert OUTPUT_INNINGS2_PATH.exists(), (
            f"Expected output file not found: {OUTPUT_INNINGS2_PATH}"
        )

    def test_deliveries_chase_not_empty(self, chase_output):
        assert OUTPUT_FULL_PATH.stat().st_size > 0

    def test_innings2_chase_not_empty(self, chase_output):
        assert OUTPUT_INNINGS2_PATH.stat().st_size > 0

# Schema tests
NEW_COLUMNS = {
    "balls_remaining", "runs_required", "required_run_rate",
    "wickets_in_hand", "over_number", "phase",
}

class TestSchema:
    """All new feature columns must be present in both output DataFrames."""

    def test_full_df_has_new_columns(self, full_df):
        missing = NEW_COLUMNS - set(full_df.columns)
        assert not missing, f"full_df missing columns: {missing}"

    def test_innings2_df_has_new_columns(self, innings2_df):
        missing = NEW_COLUMNS - set(innings2_df.columns)
        assert not missing, f"innings2_df missing columns: {missing}"

    def test_innings2_only_contains_innings2(self, innings2_df):
        """innings2_chase.csv must only contain second-innings rows."""
        bad = (innings2_df["innings"] != 2).sum()
        assert bad == 0, (
            f"innings2_df contains {bad} rows that are not innings 2"
        )

    def test_row_count_preserved_in_full_df(self, full_df):
        """Row count of full_df must match deliveries_clean.csv (295,732)."""
        assert len(full_df) == 295_732, (
            f"Expected 295,732 rows in full_df, got {len(full_df)}"
        )

    def test_innings2_row_count_positive(self, innings2_df):
        """innings2_df must have a positive number of rows."""
        assert len(innings2_df) > 0

# Numeric range sanity
class TestNumericRanges:
    """Feature values must respect cricket constraints."""

    def test_balls_remaining_non_negative(self, full_df):
        n = (full_df["balls_remaining"] < 0).sum()
        assert n == 0, f"{n} rows have balls_remaining < 0"

    def test_balls_remaining_max_120(self, full_df):
        n = (full_df["balls_remaining"] > TOTAL_LEGAL_BALLS).sum()
        assert n == 0, (
            f"{n} rows have balls_remaining > {TOTAL_LEGAL_BALLS}"
        )

    def test_wickets_in_hand_range(self, full_df):
        n_neg = (full_df["wickets_in_hand"] < 0).sum()
        n_over = (full_df["wickets_in_hand"] > 10).sum()
        assert n_neg == 0, f"{n_neg} rows have wickets_in_hand < 0"
        assert n_over == 0, f"{n_over} rows have wickets_in_hand > 10"

    def test_over_number_non_negative(self, full_df):
        """Over number must never be negative."""
        n_neg = (full_df["over_number"] < 0).sum()
        assert n_neg == 0, f"{n_neg} rows have over_number < 0"

    def test_over_number_anomaly_known_case(self, full_df):
        """
        Known anomaly: match 419155 innings 1 has one delivery with
        legal_balls_before == 120, giving over_number == 20 (the 121st
        legal ball in an innings). This was documented in the Milestone 1
        dataset audit (docs/DATASET_AUDIT.md) and is the ONLY such row.

        We assert exactly 1 such row exists. If a data update produces
        more, this test will fail, signalling a new issue.
        """
        extra_over = full_df[full_df["over_number"] > 19]
        assert len(extra_over) == 1, (
            f"Expected exactly 1 row with over_number > 19 (match 419155 anomaly), "
            f"found {len(extra_over)}:\n"
            f"{extra_over[['match_id','innings','over','legal_balls_before','over_number']].to_string()}"
        )
        row = extra_over.iloc[0]
        assert row["match_id"] == 419155, (
            f"Unexpected match for over_number > 19 anomaly: {row['match_id']}"
        )
        assert row["innings"] == 1, (
            f"Anomaly expected in innings 1, found innings {row['innings']}"
        )

    def test_rrr_non_negative_in_innings2(self, innings2_df):
        """required_run_rate must not be negative for innings 2."""
        # Use fillna(0) to ignore sentinel rows (99.0) in comparison.
        n = (innings2_df["required_run_rate"].fillna(0) < 0).sum()
        assert n == 0, f"{n} innings-2 rows have required_run_rate < 0"

    def test_no_nan_in_rrr_for_innings2(self, innings2_df):
        """required_run_rate must not be NaN for any innings-2 row."""
        n = innings2_df["required_run_rate"].isna().sum()
        assert n == 0, (
            f"{n} innings-2 rows have NaN required_run_rate "
            "(target is available for all innings-2 rows)"
        )

# Innings 1: chase features must be NaN
class TestInnings1Features:
    """First-innings chase features must be NaN (no target exists) """

    def test_runs_required_nan_for_innings1(self, full_df):
        """runs_required must be NaN for all first-innings rows """
        inn1 = full_df[full_df["innings"] == 1]
        non_nan = inn1["runs_required"].notna().sum()
        assert non_nan == 0, (
            f"{non_nan} innings-1 rows have non-NaN runs_required"
        )

    def test_rrr_nan_for_innings1(self, full_df):
        """required_run_rate must be NaN for all first-innings rows."""
        inn1 = full_df[full_df["innings"] == 1]
        non_nan = inn1["required_run_rate"].notna().sum()
        assert non_nan == 0, (
            f"{non_nan} innings-1 rows have non-NaN required_run_rate"
        )

# Phase label correctness
class TestPhaseLabels:
    """Phase labels must be consistent with over_number """

    def test_powerplay_overs(self, full_df):
        """Overs 0-5 must be labelled 'powerplay'."""
        pp = full_df[full_df["over_number"] <= 5]
        bad = (pp["phase"] != "powerplay").sum()
        assert bad == 0, f"{bad} powerplay overs have wrong phase label"

    def test_middle_overs(self, full_df):
        """Overs 6-14 must be labelled 'middle' """
        mid = full_df[(full_df["over_number"] >= 6) & (full_df["over_number"] <= 14)]
        bad = (mid["phase"] != "middle").sum()
        assert bad == 0, f"{bad} middle overs have wrong phase label"

    def test_death_overs(self, full_df):
        """Overs 15-19 must be labelled 'death'."""
        death = full_df[full_df["over_number"] >= 15]
        bad = (death["phase"] != "death").sum()
        assert bad == 0, f"{bad} death overs have wrong phase label"

    def test_only_three_phases(self, full_df):
        """There must be exactly 3 distinct phase values."""
        phases = set(full_df["phase"].unique())
        assert phases == {"powerplay", "middle", "death"}, (
            f"Unexpected phase values: {phases}"
        )

# RRR formula spot-check against manual calculation
class TestRRRFormula:
    """Verify RRR formula correctness on known rows"""

    def test_rrr_formula_correct(self, innings2_df):
        """
        For a sample of normal innings-2 rows (runs_required > 0, balls_remaining > 0),
        verify: required_run_rate ≈ (runs_required * 6) / balls_remaining.
        """
        normal_rows = innings2_df[
            (innings2_df["runs_required"] > 0)
            & (innings2_df["balls_remaining"] > 0)
        ].sample(n=min(1000, len(innings2_df)), random_state=42)

        expected = (normal_rows["runs_required"] * 6.0) / normal_rows["balls_remaining"]
        actual = normal_rows["required_run_rate"]

        max_diff = (expected - actual).abs().max()
        assert max_diff < 1e-6, (
            f"RRR formula mismatch: max absolute difference = {max_diff}"
        )

    def test_rrr_sentinel_when_no_balls_remain(self, innings2_df):
        """When balls_remaining == 0, required_run_rate must be RRR_SENTINEL"""
        no_balls = innings2_df[
            (innings2_df["balls_remaining"] == 0)
            & (innings2_df["runs_required"] > 0)
        ]
        if len(no_balls) == 0:
            pytest.skip("No rows with balls_remaining == 0 found in innings 2")
        wrong = (no_balls["required_run_rate"] != RRR_SENTINEL).sum()
        assert wrong == 0, (
            f"{wrong} rows with balls_remaining==0 have RRR != {RRR_SENTINEL}"
        )

    def test_rrr_zero_when_target_passed(self, innings2_df):
        """When runs_required <= 0, required_run_rate must be 0.0."""
        target_passed = innings2_df[innings2_df["runs_required"] <= 0]
        if len(target_passed) == 0:
            pytest.skip("No rows with runs_required <= 0 found in innings 2")
        wrong = (target_passed["required_run_rate"] != 0.0).sum()
        assert wrong == 0, (
            f"{wrong} rows with runs_required<=0 have RRR != 0.0"
        )

# Ordering test
class TestOrdering:
    """innings2_chase.csv must be sorted chronologically within each match"""

    SAMPLE_MATCH_IDS = [1082591, 598004, 1175365]

    @pytest.mark.parametrize("match_id", SAMPLE_MATCH_IDS)
    def test_legal_balls_before_sorted(self, innings2_df, match_id):
        """legal_balls_before must be non-decreasing within each match."""
        match_rows = innings2_df[innings2_df["match_id"] == match_id]
        if len(match_rows) == 0:
            pytest.skip(f"Match {match_id} not found in innings2_df")

        diffs = match_rows["legal_balls_before"].diff().dropna()
        n_bad = (diffs < 0).sum()
        assert n_bad == 0, (
            f"Match {match_id}: innings2_df has {n_bad} rows where "
            "legal_balls_before decreases (wrong sort order)"
        )
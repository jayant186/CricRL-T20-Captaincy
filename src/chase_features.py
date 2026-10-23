from pathlib import Path

import numpy as np
import pandas as pd

# Paths
ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"

INPUT_PATH = PROCESSED_DIR / "deliveries_clean.csv"
OUTPUT_FULL_PATH = PROCESSED_DIR / "deliveries_chase.csv"
OUTPUT_INNINGS2_PATH = PROCESSED_DIR / "innings2_chase.csv"

# Maximum legal balls in a standard T20 innings
TOTAL_LEGAL_BALLS = 120

# Sentinel RRR when no balls remain (avoids divide-by-zero; signals end of innings)
RRR_SENTINEL = 99.0

# Feature computation
def _compute_chase_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add six derived columns to the deliveries DataFrame.
    All arithmetic is vectorized (no Python loops) so it runs fast on 295 k rows.

    Parameters
    df : pd.DataFrame
        Deliveries data (all innings). Must contain:
        legal_balls_before, wickets_before, target, runs_required_before.

    Returns
    pd.DataFrame
        Input DataFrame with six new columns appended.
    """
    # balls_remaining 
    # Valid for all innings. For a super over it can go negative; we clamp to 0 to avoid confusion (super overs are filtered out in data_loader anyway).
    df["balls_remaining"] = (TOTAL_LEGAL_BALLS - df["legal_balls_before"]).clip(lower=0)

    # over_number 
    # Integer division of legal balls delivered gives the current over number (0-indexed).
    # Example: 12 legal balls have been bowled → we are in over 2 (overs 0 and 1 complete).
    df["over_number"] = df["legal_balls_before"] // 6

    # phase 
    # T20 phase boundaries (standard IPL definitions):
    #   Powerplay : overs 0–5   (mandatory fielding restrictions, overs 1-6)
    #   Middle    : overs 6–14  (free overs, batsmen consolidate or accelerate)
    #   Death     : overs 15–19 (final overs, maximum aggression)
    def _phase(over: int) -> str:
        if over <= 5:
            return "powerplay"
        elif over <= 14:
            return "middle"
        else:
            return "death"

    df["phase"] = df["over_number"].map(_phase)

    # runs_required 
    # This is exactly runs_required_before already in the CSV; we add it as an explicit column under the canonical name to make the feature set self-contained.
    # For innings 1, target is NaN → runs_required is NaN.
    df["runs_required"] = df["runs_required_before"]

    # wickets_in_hand
    # Complement of wickets_before. Ranges from 10 (start) down to 0 (all out).
    df["wickets_in_hand"] = 10 - df["wickets_before"]

    # required_run_rate 
    # Formula: RRR = runs_required / overs_remaining = (runs_required × 6) / balls_remaining
    # We use np.where to handle the three special cases:
    #   (a) First innings (runs_required is NaN) → NaN
    #   (b) balls_remaining == 0 → sentinel 99.0
    #   (c) runs_required <= 0  → 0.0 (already won)
    # All other rows → (runs_required * 6) / balls_remaining
    rr = df["runs_required"]
    br = df["balls_remaining"]

    # Build required_run_rate step by step to keep the logic readable.
    # Start with NaN (handles innings 1 automatically since rr is NaN there).
    rrr = pd.Series(np.nan, index=df.index, dtype="float64")

    # Rows where runs_required is a real number (innings 2 rows).
    has_target = rr.notna()

    # Among target rows: already won (runs_required <= 0) → RRR = 0.
    won = has_target & (rr <= 0)
    rrr.loc[won] = 0.0

    # Among target rows: last ball already bowled (balls_remaining == 0) → sentinel.
    last_ball = has_target & (rr > 0) & (br == 0)
    rrr.loc[last_ball] = RRR_SENTINEL

    # Remaining target rows: normal calculation.
    normal = has_target & (rr > 0) & (br > 0)
    rrr.loc[normal] = (rr.loc[normal] * 6.0) / br.loc[normal]

    df["required_run_rate"] = rrr

    return df

# Validation
def _validate_chase_features(df: pd.DataFrame) -> list[str]:
    """
    Run sanity checks on the computed chase features.

    Returns a list of warning strings. An empty list means all checks passed.
    Warnings are printed but do NOT stop execution — the user decides whether to treat them as errors.
    """
    warnings = []

    # balls_remaining must be non-negative.
    n_neg = (df["balls_remaining"] < 0).sum()
    if n_neg:
        warnings.append(f"WARN: {n_neg} rows have balls_remaining < 0")

    # balls_remaining must not exceed 120.
    n_over = (df["balls_remaining"] > 120).sum()
    if n_over:
        warnings.append(f"WARN: {n_over} rows have balls_remaining > 120")

    # wickets_in_hand must be in [0, 10].
    n_neg_wih = (df["wickets_in_hand"] < 0).sum()
    n_over_wih = (df["wickets_in_hand"] > 10).sum()
    if n_neg_wih:
        warnings.append(f"WARN: {n_neg_wih} rows have wickets_in_hand < 0")
    if n_over_wih:
        warnings.append(f"WARN: {n_over_wih} rows have wickets_in_hand > 10")

    # For innings 2: required_run_rate must not be NaN (unless it is the sentinel)
    inn2 = df[df["innings"] == 2]
    rrr_nan = inn2["required_run_rate"].isna().sum()
    if rrr_nan:
        warnings.append(
            f"WARN: {rrr_nan} innings-2 rows have NaN required_run_rate"
        )

    # RRR must be non-negative
    n_neg_rrr = (df["required_run_rate"].fillna(0) < 0).sum()
    if n_neg_rrr:
        warnings.append(f"WARN: {n_neg_rrr} rows have required_run_rate < 0")

    # phase must only take known values
    valid_phases = {"powerplay", "middle", "death"}
    bad_phases = (~df["phase"].isin(valid_phases)).sum()
    if bad_phases:
        warnings.append(
            f"WARN: {bad_phases} rows have unexpected phase value "
            f"(expected {valid_phases})"
        )

    return warnings

# Main
def build_chase_features(verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load deliveries_clean.csv, compute chase features, validate, and write outputs.

    Parameters
    verbose : bool
        If True, print a progress report.

    Returns
    (full_df, innings2_df) : tuple of DataFrames
        full_df    — all deliveries with new columns
        innings2_df — second-innings deliveries only
    """
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input not found: {INPUT_PATH}. "
            "Run src/data_processing.py and src/clean_team_names.py first."
        )

    # Load
    if verbose:
        print(f"Loading {INPUT_PATH} ...")
    df = pd.read_csv(INPUT_PATH, low_memory=False)

    if verbose:
        print(f"  Rows loaded: {len(df):,}")
        print(f"  Columns before feature engineering: {len(df.columns)}")

    # Compute features (in-place; returns the same DataFrame for chaining)
    df = _compute_chase_features(df)

    if verbose:
        print(f"  Columns after feature engineering: {len(df.columns)}")

    # Validate.
    warnings = _validate_chase_features(df)
    if warnings:
        for w in warnings:
            print(w)
    else:
        if verbose:
            print("  Validation: all checks passed.")

    # Write full output (all innings, new columns appended)
    df.to_csv(OUTPUT_FULL_PATH, index=False)
    if verbose:
        print(f"  Written: {OUTPUT_FULL_PATH}")

    # Second-innings only: filter and sort
    # Sort by match_id then legal_balls_before ensures chronological ball order within each match-innings, which the simulator relies on
    inn2 = (
        df[df["innings"] == 2]
        .copy()
        .sort_values(["match_id", "legal_balls_before"])
        .reset_index(drop=True)
    )

    inn2.to_csv(OUTPUT_INNINGS2_PATH, index=False)
    if verbose:
        print(f"  Written: {OUTPUT_INNINGS2_PATH}")
        print(f"  innings2 rows: {len(inn2):,}")
        print()

        # Quick summary of the new features
        print("=== Chase Feature Summary (innings 2 only) ===")
        print(inn2[
            ["balls_remaining", "runs_required", "required_run_rate",
             "wickets_in_hand", "over_number", "phase"]
        ].describe().round(2).to_string())
        print()
        print("Phase distribution (innings 2):")
        print(inn2["phase"].value_counts().sort_index().to_string())

    return df, inn2


def main():
    build_chase_features(verbose=True)


if __name__ == "__main__":
    main()
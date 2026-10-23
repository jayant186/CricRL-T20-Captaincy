from pathlib import Path
import numpy as np
import pandas as pd

# Paths
ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"


# Constants

# A full T20 innings requires at least 10 overs of legal balls to be considered complete enough for simulator calibration
MIN_LEGAL_BALLS_PER_INNINGS = 60

# Train fraction (80 % of valid match IDs; 20 % held out for evaluation)
# Splitting by sorted match_id preserves approximate chronological order so that the test set contains more recent seasons — a realistic deployment scenario where we train on historical data and evaluate on newer matches
TRAIN_FRACTION = 0.8

# Internal helpers
def _read_deliveries_clean() -> pd.DataFrame:
    """Load deliveries_clean.csv with correct dtype hints."""
    path = PROCESSED_DIR / "deliveries_clean.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"deliveries_clean.csv not found at {path}. "
            "Run src/data_processing.py and src/clean_team_names.py first."
        )
    df = pd.read_csv(
        path,
        # Avoid pandas guessing mixed-type columns as 'object' with low_memory
        low_memory=False,
        dtype={
            "match_id": "int64",
            "innings": "int64",
            "over": "int64",
            "delivery_in_over": "int64",
            "runs_before": "int64",
            "wickets_before": "int64",
            "legal_balls_before": "int64",
            "batter_runs": "int64",
            "extra_runs": "int64",
            "total_runs": "int64",
            "wickets_on_delivery": "int64",
            "runs_after": "int64",
            "wickets_after": "int64",
            "legal_balls_after": "int64",
        },
    )
    return df


def _read_match_summary_clean() -> pd.DataFrame:
    """Load match_summary_clean.csv."""
    path = PROCESSED_DIR / "match_summary_clean.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"match_summary_clean.csv not found at {path}. "
            "Run src/clean_team_names.py first."
        )
    return pd.read_csv(path, low_memory=False)


def _identify_valid_match_ids(match_summary: pd.DataFrame) -> set:
    """
    Return the set of match_ids that pass all three quality filters.

    A match is excluded if ANY of the following is true:
    - It has a super-over innings (innings > 2).
    - Either of its regular innings had fewer than MIN_LEGAL_BALLS_PER_INNINGS legal balls (DL/rain-affected).
    - Its winner is NaN (no result, tied, abandoned, etc.).

    Parameters
    match_summary : pd.DataFrame
        The full match_summary_clean.csv, all innings included.

    Returns
    set of int : Valid match IDs.
    """
    # Step 1: Collect all match IDs that have a super-over innings.
    super_over_match_ids = set(
        match_summary.loc[match_summary["innings"] > 2, "match_id"].unique()
    )

    # Step 2: Among regular innings only, find matches where either innings had fewer than the minimum legal-ball threshold.
    regular = match_summary[match_summary["innings"] <= 2].copy()
    # Group by match; take the minimum legal_balls across both innings of a match
    match_min_balls = regular.groupby("match_id")["legal_balls"].min()
    short_match_ids = set(
        match_min_balls[match_min_balls < MIN_LEGAL_BALLS_PER_INNINGS].index
    )

    # Step 3: Find no-result matches (winner column is NaN for at least one innings)
    no_result_match_ids = set(
        regular.loc[regular["winner"].isna(), "match_id"].unique()
    )

    excluded = super_over_match_ids | short_match_ids | no_result_match_ids

    all_regular_match_ids = set(regular["match_id"].unique())
    valid = all_regular_match_ids - excluded
    return valid


def _split_match_ids(
    match_ids: set,
    train_fraction: float = TRAIN_FRACTION,
    random_seed: int = 42,
) -> tuple[list, list]:
    """
    Split match IDs into train and test lists.

    Parameters
    match_ids : set
        Valid match IDs.
    train_fraction : float
        Fraction of matches for training (default 0.8).
    random_seed : int
        NumPy random seed for reproducibility.

    Returns
    (train_ids, test_ids) : tuple of lists
    """
    rng = np.random.default_rng(seed=random_seed)
    # Sort first so the shuffle is deterministic regardless of set iteration order
    sorted_ids = sorted(match_ids)
    rng.shuffle(sorted_ids)

    n_train = int(len(sorted_ids) * train_fraction)
    train_ids = sorted_ids[:n_train]
    test_ids = sorted_ids[n_train:]
    return train_ids, test_ids


def _verify_no_leakage(train_ids: list, test_ids: list) -> None:
    """Raise AssertionError if the same match_id appears in both splits."""
    overlap = set(train_ids) & set(test_ids)
    if overlap:
        raise AssertionError(
            f"Data leakage detected: {len(overlap)} match IDs appear in both "
            f"train and test sets. First 5: {sorted(overlap)[:5]}"
        )


# Public API

def load_pipeline_data(
    train_fraction: float = TRAIN_FRACTION,
    random_seed: int = 42,
    verbose: bool = False,
) -> dict:
    """
    Load, filter, and split the CricRL dataset.

    Returns
    dict with keys:
        "deliveries_clean"      : full deliveries_clean.csv (all matches, unfiltered)
        "match_summary_clean"   : full match_summary_clean.csv (unfiltered)
        "valid_match_ids"       : set of int — matches that pass all filters
        "train_match_ids"       : list of int — training match IDs
        "test_match_ids"        : list of int — test match IDs
        "train_deliveries"      : DataFrame — delivery rows for training matches,
                                  regular innings only (innings <= 2)
        "test_deliveries"       : DataFrame — delivery rows for test matches,
                                  regular innings only (innings <= 2)
        "train_match_summary"   : DataFrame — summary rows for training matches
        "test_match_summary"    : DataFrame — summary rows for test matches
        "filter_report"         : dict — counts at each filtering step
    """
    deliveries = _read_deliveries_clean()
    match_summary = _read_match_summary_clean()

    # Identify valid matches 
    all_match_ids = set(match_summary["match_id"].unique())
    valid_match_ids = _identify_valid_match_ids(match_summary)

    excluded = all_match_ids - valid_match_ids
    n_super_over = len(
        set(match_summary.loc[match_summary["innings"] > 2, "match_id"].unique())
        - set(match_summary.loc[match_summary["innings"] <= 2, "match_id"].unique())
    )

    # Recompute sub-categories for the report (some matches fall in multiple categories)
    regular = match_summary[match_summary["innings"] <= 2]
    match_min_balls = regular.groupby("match_id")["legal_balls"].min()
    n_short = int((match_min_balls < MIN_LEGAL_BALLS_PER_INNINGS).sum())
    n_no_result = int(regular.loc[regular["winner"].isna(), "match_id"].nunique())

    filter_report = {
        "total_match_ids": len(all_match_ids),
        "matches_with_super_over_innings": int(
            match_summary.loc[match_summary["innings"] > 2, "match_id"].nunique()
        ),
        "matches_with_short_innings": n_short,
        "no_result_matches": n_no_result,
        "total_excluded": len(excluded),
        "valid_matches": len(valid_match_ids),
    }

    # Train / test split 
    train_ids, test_ids = _split_match_ids(
        valid_match_ids,
        train_fraction=train_fraction,
        random_seed=random_seed,
    )
    _verify_no_leakage(train_ids, test_ids)

    train_set = set(train_ids)
    test_set = set(test_ids)

    # Filter delivery and summary DataFrames
    # Keep only regular innings (innings <= 2) for the ML pipeline.
    regular_deliveries = deliveries[deliveries["innings"] <= 2].copy()

    train_deliveries = regular_deliveries[
        regular_deliveries["match_id"].isin(train_set)
    ].reset_index(drop=True)

    test_deliveries = regular_deliveries[
        regular_deliveries["match_id"].isin(test_set)
    ].reset_index(drop=True)

    regular_summary = match_summary[match_summary["innings"] <= 2].copy()

    train_match_summary = regular_summary[
        regular_summary["match_id"].isin(train_set)
    ].reset_index(drop=True)

    test_match_summary = regular_summary[
        regular_summary["match_id"].isin(test_set)
    ].reset_index(drop=True)

    if verbose:
        print("=== load_pipeline_data() filter report ===")
        for k, v in filter_report.items():
            print(f"  {k}: {v}")
        print(f"  train_match_ids: {len(train_ids)}")
        print(f"  test_match_ids:  {len(test_ids)}")
        print(f"  train_deliveries rows: {len(train_deliveries)}")
        print(f"  test_deliveries rows:  {len(test_deliveries)}")

    return {
        "deliveries_clean": deliveries,
        "match_summary_clean": match_summary,
        "valid_match_ids": valid_match_ids,
        "train_match_ids": train_ids,
        "test_match_ids": test_ids,
        "train_deliveries": train_deliveries,
        "test_deliveries": test_deliveries,
        "train_match_summary": train_match_summary,
        "test_match_summary": test_match_summary,
        "filter_report": filter_report,
    }
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Union

import numpy as np
import pandas as pd

# MatchState dataclass
@dataclass
class MatchState:
    """
    Snapshot of the match situation *before* a given delivery is bowled.

    All fields correspond to second-innings chase situations.
    First-innings rows can be converted but `target`, `runs_required`, and
    `required_run_rate` will be NaN/sentinel values.

    Attributes
    current_score : int
        Runs scored by the batting team so far in this innings
    wickets_lost : int
        Wickets fallen so far (0-10). At 10 the innings is over
    balls_remaining : int
        Legal balls remaining in the innings: 120 - legal_balls_before
    target : float
        Runs needed to win (first innings total + 1). NaN for innings 1
    runs_required : float
        Runs still needed: target - current_score. NaN for innings 1
    required_run_rate : float
        Runs per over (6 balls) needed to win
        = (runs_required * 6) / balls_remaining
        Sentinel 99.0 when balls_remaining == 0
        NaN for innings 1
    """

    current_score: int
    wickets_lost: int
    balls_remaining: int
    target: float          # NaN for innings 1
    runs_required: float   # NaN for innings 1
    required_run_rate: float  # NaN for innings 1 or sentinel 99.0

    # Max-value constants used for normalization
    # These are conservative upper bounds; clipping ensures robustness
    _SCORE_MAX: float = 250.0
    _WICKETS_MAX: float = 10.0
    _BALLS_MAX: float = 120.0
    _TARGET_MAX: float = 300.0
    _RUNS_REQ_MAX: float = 300.0
    _RRR_MAX: float = 36.0   # 6 runs/ball * 6 balls/over

    # Sentinel used when balls_remaining == 0 (innings ended, RRR undefined)
    _RRR_SENTINEL: float = 99.0


# Constructor from a raw CSV row
def state_from_delivery_row(row: Union[pd.Series, dict]) -> MatchState:
    """
    Build a MatchState from one row of deliveries_clean.csv.

    The row must contain these columns (present in deliveries_clean.csv):
        match_id, innings, legal_balls_before, runs_before, wickets_before, target, runs_required_before

    Parameters
    row : pd.Series or dict
        A single delivery record.

    Returns
    MatchState
        State *before* this delivery is bowled.

    Raises
    KeyError
        If a required column is missing from the row.
    ValueError
        If the computed balls_remaining is negative (data integrity error).
    """
    # Required columns 
    legal_balls_before = int(row["legal_balls_before"])
    runs_before = int(row["runs_before"])
    wickets_before = int(row["wickets_before"])

    # target and runs_required_before are NaN for innings 1
    raw_target = row["target"]
    raw_runs_req = row["runs_required_before"]

    # balls_remaining 
    # A standard T20 innings has exactly 120 legal balls (20 overs × 6 balls)
    # legal_balls_before counts how many legal balls have been bowled so far
    balls_remaining = 120 - legal_balls_before
    if balls_remaining < 0:
        raise ValueError(
            f"balls_remaining={balls_remaining} is negative for row with "
            f"legal_balls_before={legal_balls_before}. "
            "This indicates data beyond 120 legal balls (super over or DL match)."
        )

    # target and runs_required
    target = float(raw_target) if not _is_na(raw_target) else math.nan
    runs_required = float(raw_runs_req) if not _is_na(raw_runs_req) else math.nan

    # required_run_rate 
    # Formula: RRR = runs_required/overs_remaining = (runs_required * 6) / balls_remaining
    # The factor of 6 converts "balls remaining" into "overs remaining" so the result is expressed as runs-per-over (the standard cricket unit)
    if math.isnan(runs_required):
        required_run_rate = math.nan
    elif balls_remaining == 0:
        # Innings ended; RRR is undefined (divide-by-zero)
        required_run_rate = MatchState._RRR_SENTINEL
    elif runs_required <= 0:
        # Target already passed (batting team won)
        required_run_rate = 0.0
    else:
        required_run_rate = (runs_required * 6.0) /balls_remaining

    return MatchState(
        current_score=runs_before,
        wickets_lost=wickets_before,
        balls_remaining=balls_remaining,
        target=target,
        runs_required=runs_required,
        required_run_rate=required_run_rate,
    )

# Conversion to normalized observation vector

def state_to_observation(state: MatchState) -> np.ndarray:
    """
    Convert a MatchState to a normalized numpy float32 vector of length 6

    This is the tensor that will be fed directly into the DQN Q-network

    Normalization divides each feature by its known maximum value, then clips to [0, 1]. Clipping handles the rare edge cases (e.g., a target > 300) without crashing
    Feature order (fixed; changing this breaks saved model checkpoints):
        [0] current_score      / 250
        [1] wickets_lost       / 10
        [2] balls_remaining    / 120
        [3] target             / 300  (0.0 for innings 1, where target is NaN)
        [4] runs_required      / 300  (0.0 for innings 1)
        [5] required_run_rate  / 36   (0.0 for innings 1)

    Parameters
    state : MatchState

    Returns
    np.ndarray of shape (6,) and dtype float32
    """
    ms = state  # shorter alias

    def _safe(value: float, max_val: float) -> float:
        """Normalize and clip; treat NaN as 0.0."""
        if math.isnan(value):
            return 0.0
        return float(np.clip(value / max_val, 0.0, 1.0))

    obs = np.array(
        [
            _safe(ms.current_score, ms._SCORE_MAX),
            _safe(ms.wickets_lost, ms._WICKETS_MAX),
            _safe(ms.balls_remaining, ms._BALLS_MAX),
            _safe(ms.target, ms._TARGET_MAX),
            _safe(ms.runs_required, ms._RUNS_REQ_MAX),
            _safe(ms.required_run_rate, ms._RRR_MAX),
        ],
        dtype=np.float32,
    )
    return obs

# Internal helpers
def _is_na(value) -> bool:
    """Return True if value is NaN, None, or pandas NA"""
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return value is None or (hasattr(pd, "isna") and pd.isna(value))
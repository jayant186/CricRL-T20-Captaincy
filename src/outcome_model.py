"""
src/outcome_model.py — Milestone 3: Delivery Outcome Model
===========================================================

Purpose
-------
Estimate the probability distribution over delivery outcomes from historical
IPL second-innings data. The simulator draws from these distributions to
generate synthetic deliveries; it does NOT replay historical deliveries.

Why a statistical model instead of direct replay?
--------------------------------------------------
Direct replay would produce exactly one outcome per historical delivery,
meaning the agent would always see the same sequence of balls for a given
match state. The simulator needs to be stochastic so that:
1. The agent explores different futures from the same state.
2. Training averages over many rollouts, reducing variance in policy gradients.
3. We can simulate more matches than exist in the dataset.

Model design
------------
We use a **conditioned empirical frequency table** (no parametric assumptions).
For each cell of (phase × wickets_in_hand_bucket), we store:

    P(batter_runs = k | phase, wih_bucket)     for k in {0,1,2,3,4,5,6}
    P(wicket | phase, wih_bucket)              on legal deliveries only
    P(wide | phase)                            delivery-level
    P(no-ball | phase)                         delivery-level

Conditioning on phase and wickets_in_hand captures:
- Phase (powerplay/middle/death): fielding restrictions, match pressure, and
  typical bowling aggression all change significantly by phase.
- Wickets in hand: batting team's resources. Teams with few wickets play more
  conservatively (more 0s and 1s, fewer 6s).

Aggression modifiers
--------------------
The DQN agent's three actions are:
    0 = Conservative  (prefer singles and twos; avoid risk)
    1 = Balanced      (historical average probabilities, no modification)
    2 = Aggressive    (prefer boundaries; accept higher wicket risk)

We implement aggression as a multiplicative reweighting of the empirical
probabilities. This is a modelling assumption (not directly observable in data)
but is the standard approach in T20 captaincy simulation papers.

    Conservative: multiply P(4) and P(6) by 0.6, P(wicket) by 0.75,
                  renormalize batter_runs distribution.
    Balanced:     use empirical probabilities unchanged.
    Aggressive:   multiply P(4) and P(6) by 1.5, P(wicket) by 1.30,
                  renormalize batter_runs distribution.

These multipliers are documented assumptions, not calibrated values.
They produce qualitatively sensible behavior (aggressive → more 4s/6s/wickets)
while keeping the model simple and interpretable.

Wicket and batter_runs joint sampling
--------------------------------------
Wickets and batter_runs are *not* fully independent in the real data:
  - When a wicket falls: batter_runs = 0 in 97.5% of deliveries
    (bowled, caught, LBW, stumped typically score 0).
  - The remaining 2.5% are run-out situations (1 or 2 runs taken before runout).

Previous code sampled batter_runs and is_wicket independently, which produced
boundary+wicket combinations (~22% of wicket deliveries with aggressive batting)
that are near-impossible in real cricket.

Fix: when is_wicket is sampled True, batter_runs is forced to 0. This matches
the 97.5% empirical rate and is documented as a simplifying assumption (ignoring
the rare 2.5% run-out singles/twos in favour of a clean, conservative approximation).

Data used
---------
Only training-split matches (defined by data_loader.load_pipeline_data()) are
used to fit the outcome model. Test matches are never touched.

Usage
-----
    from src.outcome_model import OutcomeModel
    model = OutcomeModel()
    model.fit(train_innings2_df)

    # Sample one delivery outcome at a given state
    outcome = model.sample(phase='death', wickets_in_hand=4,
                           aggression=2, rng=rng)
    # outcome.batter_runs, outcome.is_wicket, outcome.is_wide, outcome.is_noball
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Possible batter runs values in cricket T20.
BATTER_RUNS_VALUES = [0, 1, 2, 3, 4, 5, 6]

# Wickets-in-hand bucket boundaries.
# Bucket labels: '1-2' means 1 or 2 wickets in hand (8 or 9 wickets fallen).
WIH_BINS   = [0, 2, 4, 6, 8, 10]
WIH_LABELS = ["1-2", "3-4", "5-6", "7-8", "9-10"]

# Valid aggression actions and their boundary-probability multipliers.
# Conservative: 60 % of boundary probability, 75 % of wicket probability.
# Aggressive: 150 % of boundary probability, 130 % of wicket probability.
AGGRESSION_BOUNDARY_MULT = {0: 0.60, 1: 1.00, 2: 1.50}
AGGRESSION_WICKET_MULT   = {0: 0.75, 1: 1.00, 2: 1.30}

# Minimum observations required per cell before we use a fallback distribution.
MIN_CELL_OBS = 20

# Extra runs added for a wide (1) or no-ball (1) — the standard ICC rule.
WIDE_EXTRA_RUNS = 1
NOBALL_EXTRA_RUNS = 1


# ─────────────────────────────────────────────────────────────────────────────
# DeliveryOutcome dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DeliveryOutcome:
    """
    The result of a single simulated delivery.

    Attributes
    ----------
    batter_runs : int
        Runs credited to the batter (0–6).  On a wide, this is always 0
        because the batter does not face the ball.
    extra_runs : int
        Additional runs from extras (wide=1, no-ball=1, both can combine).
    total_runs : int
        batter_runs + extra_runs (what the scoreboard shows).
    is_wicket : bool
        True if a wicket fell on this delivery.
        On a wide or no-ball, wickets are still possible (run-outs, stumped on
        wide, etc.) but are rare. We model them as zero probability on extras
        for simplicity — a documented limitation.
    is_wide : bool
        True if the delivery is a wide.
    is_noball : bool
        True if the delivery is a no-ball.
    is_legal : bool
        True if the delivery counts toward the legal-ball count (not a wide
        or no-ball).
    """
    batter_runs: int
    extra_runs: int
    total_runs: int
    is_wicket: bool
    is_wide: bool
    is_noball: bool
    is_legal: bool


# ─────────────────────────────────────────────────────────────────────────────
# OutcomeModel
# ─────────────────────────────────────────────────────────────────────────────

class OutcomeModel:
    """
    Conditioned empirical frequency table for T20 delivery outcomes.

    After calling fit(), the model supports sample() to draw one delivery
    outcome given the current match state and aggression action.

    Attributes (set by fit())
    -------------------------
    _batter_run_probs : dict
        {(phase, wih_bucket): np.ndarray of shape (7,)} —
        empirical probability of each batter_runs value [0,1,2,3,4,5,6]
        on legal deliveries in that cell.
    _wicket_rates : dict
        {(phase, wih_bucket): float} —
        P(wicket | legal delivery, phase, wih_bucket).
    _wide_rates : dict
        {phase: float} — P(wide | phase).
    _noball_rates : dict
        {phase: float} — P(no-ball | phase).
    _fallback_probs : np.ndarray of shape (7,)
        Global empirical distribution used when a cell has < MIN_CELL_OBS rows.
    _is_fitted : bool
    """

    def __init__(self):
        self._batter_run_probs: dict = {}
        self._wicket_rates: dict = {}
        self._wide_rates: dict = {}
        self._noball_rates: dict = {}
        self._fallback_probs: Optional[np.ndarray] = None
        self._fallback_wicket_rate: float = 0.05
        self._is_fitted: bool = False

    # ─────────────────────────────────────────────────────────────────────────
    # Fitting
    # ─────────────────────────────────────────────────────────────────────────

    def fit(self, innings2_df: pd.DataFrame) -> "OutcomeModel":
        """
        Fit the outcome model from training-split second-innings delivery data.

        Parameters
        ----------
        innings2_df : pd.DataFrame
            Rows from innings2_chase.csv filtered to training match IDs.
            Must contain: phase, wickets_in_hand (or wickets_before),
            legal_delivery, batter_runs, wickets_on_delivery, extras_type.

        Returns
        -------
        self
        """
        df = innings2_df.copy()

        # Compute wickets_in_hand if missing (it's in innings2_chase.csv already).
        if "wickets_in_hand" not in df.columns:
            df["wickets_in_hand"] = 10 - df["wickets_before"]

        # Bucket wickets_in_hand.
        df["wih_bucket"] = pd.cut(
            df["wickets_in_hand"],
            bins=WIH_BINS,
            labels=WIH_LABELS,
            include_lowest=True,
        ).astype(str)

        # ── Legal-delivery subset ──────────────────────────────────────────
        legal = df[df["legal_delivery"] == True].copy()

        # ── Compute global fallback from all legal deliveries ──────────────
        self._fallback_probs = self._compute_batter_probs(legal)
        self._fallback_wicket_rate = float(legal["wickets_on_delivery"].mean())

        # ── Per-cell batter_runs probabilities and wicket rates ────────────
        for phase in ["powerplay", "middle", "death"]:
            for wib in WIH_LABELS:
                cell = legal[(legal["phase"] == phase) & (legal["wih_bucket"] == wib)]
                key = (phase, wib)
                if len(cell) >= MIN_CELL_OBS:
                    self._batter_run_probs[key] = self._compute_batter_probs(cell)
                    self._wicket_rates[key] = float(cell["wickets_on_delivery"].mean())
                else:
                    # Sparse cell — use fallback.
                    self._batter_run_probs[key] = self._fallback_probs.copy()
                    self._wicket_rates[key] = self._fallback_wicket_rate

        # ── Wide and no-ball rates by phase ───────────────────────────────
        # Computed on all deliveries (not just legal) because wides/no-balls
        # are by definition *not* legal deliveries.
        for phase in ["powerplay", "middle", "death"]:
            phase_df = df[df["phase"] == phase]
            n = len(phase_df)
            n_wide   = phase_df["extras_type"].str.contains("wides",   na=False).sum()
            n_noball = phase_df["extras_type"].str.contains("noballs", na=False).sum()
            self._wide_rates[phase]   = float(n_wide   / n) if n > 0 else 0.03
            self._noball_rates[phase] = float(n_noball / n) if n > 0 else 0.003

        self._is_fitted = True
        return self

    @staticmethod
    def _compute_batter_probs(df: pd.DataFrame) -> np.ndarray:
        """
        Compute a probability vector over BATTER_RUNS_VALUES from a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Subset of legal deliveries; must have 'batter_runs' column.

        Returns
        -------
        np.ndarray of shape (7,), summing to 1.0.
        """
        counts = np.zeros(len(BATTER_RUNS_VALUES), dtype=float)
        for i, v in enumerate(BATTER_RUNS_VALUES):
            counts[i] = (df["batter_runs"] == v).sum()
        total = counts.sum()
        if total == 0:
            # Uniform fallback — should never happen in practice.
            return np.ones(len(BATTER_RUNS_VALUES)) / len(BATTER_RUNS_VALUES)
        return counts / total

    # ─────────────────────────────────────────────────────────────────────────
    # Sampling
    # ─────────────────────────────────────────────────────────────────────────

    def sample(
        self,
        phase: str,
        wickets_in_hand: int,
        aggression: int,
        rng: np.random.Generator,
        bowling_aggression: int = 1,
    ) -> DeliveryOutcome:
        """
        Sample one delivery outcome given the current match context.

        Parameters
        ----------
        phase : str
            'powerplay', 'middle', or 'death'.
        wickets_in_hand : int
            Current wickets remaining (0–10).
        aggression : int
            Batting aggression: 0=Conservative, 1=Balanced, 2=Aggressive.
        rng : np.random.Generator
            NumPy random generator for reproducibility.
        bowling_aggression : int, default 1
            Bowling tactic: 0=Defensive, 1=Balanced, 2=Attacking.

        Returns
        -------
        DeliveryOutcome
        """
        if not self._is_fitted:
            raise RuntimeError("OutcomeModel.fit() must be called before sample().")

        # ── Step 1: Delivery type ──────────────────────────────────────────
        u = rng.random()
        wide_rate   = self._wide_rates.get(phase, 0.03)
        noball_rate = self._noball_rates.get(phase, 0.003)

        if u < wide_rate:
            return DeliveryOutcome(
                batter_runs=0,
                extra_runs=WIDE_EXTRA_RUNS,
                total_runs=WIDE_EXTRA_RUNS,
                is_wicket=False,
                is_wide=True,
                is_noball=False,
                is_legal=False,
            )

        elif u < wide_rate + noball_rate:
            key = self._cell_key(phase, wickets_in_hand)
            base_probs = self._batter_run_probs.get(key, self._fallback_probs)
            batter_runs = int(
                rng.choice(BATTER_RUNS_VALUES, p=base_probs)
            )
            return DeliveryOutcome(
                batter_runs=batter_runs,
                extra_runs=NOBALL_EXTRA_RUNS,
                total_runs=batter_runs + NOBALL_EXTRA_RUNS,
                is_wicket=False,
                is_wide=False,
                is_noball=True,
                is_legal=False,
            )

        else:
            # ── Normal (legal) delivery ────────────────────────────────────
            key = self._cell_key(phase, wickets_in_hand)
            base_probs   = self._batter_run_probs.get(key, self._fallback_probs)
            base_wkt_rate = self._wicket_rates.get(key, self._fallback_wicket_rate)

            # Apply joint batting and bowling aggression modifier to batter_runs probabilities.
            mod_probs = self._apply_aggression(base_probs, aggression, bowling_aggression)

            # Sample batter runs.
            batter_runs = int(rng.choice(BATTER_RUNS_VALUES, p=mod_probs))

            # Joint batting + bowling wicket probability
            wkt_mult = AGGRESSION_WICKET_MULT.get(aggression, 1.0)
            bowl_wkt_mult = 1.0
            if bowling_aggression == 0:    # Defensive
                bowl_wkt_mult = 0.88
            elif bowling_aggression == 2:  # Attacking
                bowl_wkt_mult = 1.25

            adjusted_wkt_rate = min(base_wkt_rate * wkt_mult * bowl_wkt_mult, 0.999)
            is_wicket = bool(rng.random() < adjusted_wkt_rate)

            if is_wicket:
                batter_runs = 0

            return DeliveryOutcome(
                batter_runs=batter_runs,
                extra_runs=0,
                total_runs=batter_runs,
                is_wicket=is_wicket,
                is_wide=False,
                is_noball=False,
                is_legal=True,
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _cell_key(phase: str, wickets_in_hand: int) -> tuple:
        """
        Map (phase, wickets_in_hand) to the lookup key used in the tables.
        """
        wih = max(1, min(10, int(wickets_in_hand)))
        for low, high, label in zip(WIH_BINS[:-1], WIH_BINS[1:], WIH_LABELS):
            if low < wih <= high or (low == 0 and wih == 0):
                return (phase, label)
        return (phase, WIH_LABELS[0])

    @staticmethod
    def _apply_aggression(base_probs: np.ndarray, aggression: int, bowling_aggression: int = 1) -> np.ndarray:
        """
        Reweight batter_runs probabilities by the joint batting and bowling modifier.
        """
        if aggression == 1 and bowling_aggression == 1:
            return base_probs.copy()

        bat_mult = AGGRESSION_BOUNDARY_MULT.get(aggression, 1.0)
        bowl_mult = 1.0
        if bowling_aggression == 0:    # Defensive
            bowl_mult = 0.80
        elif bowling_aggression == 2:  # Attacking
            bowl_mult = 1.15

        mult = bat_mult * bowl_mult
        probs = base_probs.copy()

        probs[4] *= mult  # 4s
        probs[6] *= mult  # 6s

        total = probs.sum()
        if total <= 0:
            return base_probs.copy()
        return probs / total

    # ─────────────────────────────────────────────────────────────────────────
    # Diagnostics
    # ─────────────────────────────────────────────────────────────────────────

    def summary(self) -> pd.DataFrame:
        """
        Return a DataFrame summarising all fitted cells.

        Useful for validating that distributions look sensible.
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted yet.")
        rows = []
        for (phase, wib), probs in self._batter_run_probs.items():
            wkt_rate = self._wicket_rates.get((phase, wib), float("nan"))
            row = {
                "phase": phase,
                "wih_bucket": wib,
                "wicket_rate": round(wkt_rate, 4),
            }
            for i, k in enumerate(BATTER_RUNS_VALUES):
                row[f"p_runs_{k}"] = round(float(probs[i]), 4)
            rows.append(row)
        df = pd.DataFrame(rows).sort_values(["phase", "wih_bucket"]).reset_index(drop=True)
        return df

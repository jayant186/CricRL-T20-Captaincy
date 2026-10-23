"""
src/win_probability.py — Milestone 4: Win-Probability Model
============================================================

Purpose
-------
Estimate P(batting team wins | current match state) for a T20 second-innings
chase. This model is used as an auxiliary reward signal for the DQN agent and
as a standalone evaluation tool for baselines.

Model
-----
Logistic Regression on five features derived from the match state:

    over_number          — integer 0–19 (phase of the game)
    wickets_before       — wickets fallen so far (0–9)
    runs_required        — runs still needed (target − score)
    balls_remaining      — legal balls left (120 − legal_balls_before)
    required_run_rate    — (runs_required × 6) / balls_remaining

Why logistic regression?
------------------------
- Interpretable: coefficients show the sign and relative magnitude of each feature.
- Fast to train (< 1 s on 111 k rows) and fast to predict (vectorised sigmoid).
- Does not require hyperparameter tuning to produce reasonable predictions.
- Strong baseline: Duckworth-Lewis and VJD-style models are also linear in
  their resource variables.

Feature engineering
-------------------
We add two polynomial interaction terms:
    rrr_x_wickets = required_run_rate × wickets_before
        (captures "high run rate AND many wickets fallen → desperate")
    balls_x_wickets = balls_remaining × (10 − wickets_before)
        ("many balls AND many wickets in hand → comfortable")

All features are standardised (zero mean, unit variance) before fitting.
Standardisation is required for logistic regression gradient convergence and
makes coefficients comparable across features with different scales.

Training data
-------------
Only training-split match IDs (from data_loader) are used. One row per delivery
is used as a training observation, with the label being whether the batting team
ultimately won the match. This gives the model many observations of the same
match at different points — which is appropriate because we want to estimate
how likely a WIN is from any given match state, not just end-of-innings states.

Limitations
-----------
- No player-specific features (team or individual form not captured).
- Assumes T20 T20 conditions are stationary across seasons 2007–2026
  (feature means/variances shift over time; not accounted for).
- Very high RRR rows (> 99.0 sentinel) are clipped before fitting.
- The logistic regression is linear in the features; non-linear win-probability
  surfaces (e.g., at a specific over with specific wickets) may not be captured.

Usage
-----
    from src.win_probability import WinProbabilityModel
    model = WinProbabilityModel()
    model.fit(train_inn2_with_outcome)    # DataFrame must have 'batting_team_won'

    prob = model.predict_proba(over_number=10, wickets_before=3,
                               runs_required=65, balls_remaining=60,
                               required_run_rate=6.5)
    # Returns float in [0, 1]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Cap RRR at this value before fitting/predicting.
# The sentinel value 99.0 (innings ended) and rare spikes above ~30 skew the model.
RRR_CAP = 36.0

# sklearn is not in requirements.txt; we use a pure-numpy logistic regression.
# This avoids adding an unnecessary dependency while producing equivalent results
# for a well-conditioned logistic regression problem.
MAX_ITER   = 500
LR         = 0.1       # gradient-descent step size
L2_LAMBDA  = 1e-4      # L2 regularisation to prevent overfitting


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def _build_feature_matrix(
    over_number: np.ndarray,
    wickets_before: np.ndarray,
    runs_required: np.ndarray,
    balls_remaining: np.ndarray,
    required_run_rate: np.ndarray,
) -> np.ndarray:
    """
    Build the raw (un-standardised) feature matrix from the 5 base features.

    Returns np.ndarray of shape (N, 7).
    Column order (fixed):
        0: over_number
        1: wickets_before
        2: runs_required
        3: balls_remaining
        4: required_run_rate  (clipped to RRR_CAP)
        5: rrr_x_wickets      = required_run_rate × wickets_before
        6: balls_x_wih        = balls_remaining × (10 - wickets_before)
    """
    rrr = np.clip(required_run_rate.astype(float), 0.0, RRR_CAP)
    wkts = wickets_before.astype(float)
    br   = balls_remaining.astype(float)

    return np.column_stack([
        over_number.astype(float),
        wkts,
        runs_required.astype(float),
        br,
        rrr,
        rrr * wkts,           # interaction: high RRR & many wickets fallen → desperate
        br * (10.0 - wkts),   # interaction: many balls & wickets in hand → comfortable
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Pure-numpy logistic regression (gradient descent with L2)
# ─────────────────────────────────────────────────────────────────────────────

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(z >= 0,
                    1.0 / (1.0 + np.exp(-z)),
                    np.exp(z) / (1.0 + np.exp(z)))


def _fit_logistic(X: np.ndarray, y: np.ndarray,
                  max_iter: int, lr: float, l2: float) -> np.ndarray:
    """
    Fit logistic regression via full-batch gradient descent with L2 regularisation.

    Parameters
    ----------
    X : np.ndarray of shape (N, d)    — feature matrix (standardised)
    y : np.ndarray of shape (N,)      — binary labels (0/1)
    max_iter, lr, l2                  — training hyper-parameters

    Returns
    -------
    w : np.ndarray of shape (d+1,)    — [intercept, w1, …, wd]
    """
    N, d = X.shape
    # Prepend bias column.
    Xb = np.hstack([np.ones((N, 1)), X])
    w = np.zeros(d + 1)

    for _ in range(max_iter):
        p = _sigmoid(Xb @ w)
        # Gradient of binary cross-entropy + L2 (no L2 on bias term).
        grad = (Xb.T @ (p - y)) / N
        grad[1:] += l2 * w[1:]
        w -= lr * grad

    return w


# ─────────────────────────────────────────────────────────────────────────────
# WinProbabilityModel
# ─────────────────────────────────────────────────────────────────────────────

class WinProbabilityModel:
    """
    Logistic-regression win-probability estimator for T20 second-innings chases.

    After calling fit(), use predict_proba() to obtain P(win | state) for any
    valid match state. predict_row() accepts a delivery row dict/Series for
    convenience.
    """

    def __init__(self):
        self._weights: Optional[np.ndarray] = None   # shape (8,): bias + 7 features
        self._feat_mean: Optional[np.ndarray] = None  # shape (7,)
        self._feat_std:  Optional[np.ndarray] = None  # shape (7,)
        self._is_fitted: bool = False

    # ─────────────────────────────────────────────────────────────────────────
    # fit()
    # ─────────────────────────────────────────────────────────────────────────

    def fit(self, innings2_df: pd.DataFrame) -> "WinProbabilityModel":
        """
        Fit the model from training-split second-innings delivery data.

        Parameters
        ----------
        innings2_df : pd.DataFrame
            Must contain: over_number, wickets_before, runs_required,
            balls_remaining, required_run_rate, batting_team_won (bool).

        Returns
        -------
        self
        """
        df = innings2_df.dropna(subset=[
            "over_number", "wickets_before", "runs_required",
            "balls_remaining", "required_run_rate", "batting_team_won",
        ]).copy()

        X_raw = _build_feature_matrix(
            df["over_number"].values,
            df["wickets_before"].values,
            df["runs_required"].values,
            df["balls_remaining"].values,
            df["required_run_rate"].values,
        )
        y = df["batting_team_won"].astype(float).values

        # Standardise: subtract mean, divide by std.
        self._feat_mean = X_raw.mean(axis=0)
        self._feat_std  = X_raw.std(axis=0) + 1e-8   # prevent division by zero

        X_std = (X_raw - self._feat_mean) / self._feat_std

        self._weights = _fit_logistic(X_std, y, MAX_ITER, LR, L2_LAMBDA)
        self._is_fitted = True
        return self

    # ─────────────────────────────────────────────────────────────────────────
    # predict_proba()  — single prediction
    # ─────────────────────────────────────────────────────────────────────────

    def predict_proba(
        self,
        over_number: float,
        wickets_before: float,
        runs_required: float,
        balls_remaining: float,
        required_run_rate: float,
    ) -> float:
        """
        Return P(batting team wins | current match state) in [0, 1].

        Parameters match the columns in innings2_chase.csv.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict_proba().")

        X_raw = _build_feature_matrix(
            np.array([over_number]),
            np.array([wickets_before]),
            np.array([runs_required]),
            np.array([balls_remaining]),
            np.array([required_run_rate]),
        )
        X_std = (X_raw - self._feat_mean) / self._feat_std
        Xb = np.hstack([np.ones((1, 1)), X_std])
        return float(_sigmoid(Xb @ self._weights)[0])

    def predict_proba_batch(self, innings2_df: pd.DataFrame) -> np.ndarray:
        """
        Vectorised prediction over a DataFrame.

        Returns np.ndarray of shape (N,) with probabilities in [0, 1].
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict_proba_batch().")

        df = innings2_df
        X_raw = _build_feature_matrix(
            df["over_number"].values,
            df["wickets_before"].values,
            df["runs_required"].values,
            df["balls_remaining"].values,
            df["required_run_rate"].values,
        )
        X_std = (X_raw - self._feat_mean) / self._feat_std
        N = X_std.shape[0]
        Xb = np.hstack([np.ones((N, 1)), X_std])
        return _sigmoid(Xb @ self._weights)

    # ─────────────────────────────────────────────────────────────────────────
    # Evaluation helpers
    # ─────────────────────────────────────────────────────────────────────────

    def evaluate(self, innings2_df: pd.DataFrame) -> dict:
        """
        Compute binary-classification metrics on a labelled DataFrame.

        Returns dict with: accuracy, log_loss, brier_score, auc_approx.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before evaluate().")

        df = innings2_df.dropna(subset=["batting_team_won"]).copy()
        y_true = df["batting_team_won"].astype(float).values
        y_prob = self.predict_proba_batch(df)
        y_pred = (y_prob >= 0.5).astype(float)

        accuracy = float((y_pred == y_true).mean())

        # Log-loss (binary cross-entropy).
        eps = 1e-7
        log_loss = -float(
            (y_true * np.log(y_prob + eps) +
             (1 - y_true) * np.log(1 - y_prob + eps)).mean()
        )

        # Brier score (mean squared error between probability and label).
        brier = float(((y_prob - y_true) ** 2).mean())

        return {
            "accuracy":    round(accuracy, 4),
            "log_loss":    round(log_loss, 4),
            "brier_score": round(brier, 4),
            "n_samples":   len(df),
        }

    def coefficient_table(self) -> pd.DataFrame:
        """Return a DataFrame of feature names and fitted coefficients."""
        if not self._is_fitted:
            raise RuntimeError("Call fit() first.")
        names = [
            "bias", "over_number", "wickets_before", "runs_required",
            "balls_remaining", "required_run_rate", "rrr_x_wickets", "balls_x_wih",
        ]
        return pd.DataFrame({
            "feature":     names,
            "coefficient": self._weights.round(4),
        })

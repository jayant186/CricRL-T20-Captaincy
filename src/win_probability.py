

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
   

    def __init__(self):
        self._weights: Optional[np.ndarray] = None   # shape (8,): bias + 7 features
        self._feat_mean: Optional[np.ndarray] = None  # shape (7,)
        self._feat_std:  Optional[np.ndarray] = None  # shape (7,)
        self._is_fitted: bool = False

    # ─────────────────────────────────────────────────────────────────────────
    # fit()
    # ─────────────────────────────────────────────────────────────────────────

    def fit(self, innings2_df: pd.DataFrame) -> "WinProbabilityModel":
      
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

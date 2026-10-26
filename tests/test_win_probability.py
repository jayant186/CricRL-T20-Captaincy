"""
tests/test_win_probability.py — Milestone 4: Win-Probability Model Tests
=========================================================================

Run with:
    pytest tests/test_win_probability.py -v
"""

import sys
from pathlib import Path
import math
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from win_probability import WinProbabilityModel, _build_feature_matrix, _sigmoid

DATA = Path(__file__).resolve().parents[1] / "data" / "processed"


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def train_inn2_with_outcome():
    from data_loader import load_pipeline_data
    data = load_pipeline_data(verbose=False)
    train_ids = set(data["train_match_ids"])
    test_ids  = set(data["test_match_ids"])

    inn2 = pd.read_csv(DATA / "innings2_chase.csv", low_memory=False)
    ms   = pd.read_csv(DATA / "match_summary_clean.csv", low_memory=False)

    # Build per-match batting_team_won label from innings-2 summary rows.
    ms2 = ms[ms["innings"] == 2][["match_id", "batting_team_canonical", "winner"]].copy()
    ms2["batting_team_won"] = ms2["batting_team_canonical"] == ms2["winner"]

    inn2 = inn2.merge(ms2[["match_id", "batting_team_won"]], on="match_id", how="left")

    train = inn2[inn2["match_id"].isin(train_ids)].copy()
    test  = inn2[inn2["match_id"].isin(test_ids)].copy()
    return train, test


@pytest.fixture(scope="session")
def fitted_wp_model(train_inn2_with_outcome):
    train, _ = train_inn2_with_outcome
    model = WinProbabilityModel()
    model.fit(train)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 1. Fitting tests
# ─────────────────────────────────────────────────────────────────────────────

class TestWinProbFit:

    def test_is_fitted(self, fitted_wp_model):
        assert fitted_wp_model._is_fitted

    def test_weights_shape(self, fitted_wp_model):
        # 7 features + 1 bias = 8
        assert fitted_wp_model._weights.shape == (8,)

    def test_feat_mean_shape(self, fitted_wp_model):
        assert fitted_wp_model._feat_mean.shape == (7,)

    def test_feat_std_positive(self, fitted_wp_model):
        assert np.all(fitted_wp_model._feat_std > 0)

    def test_unfitted_model_raises(self):
        model = WinProbabilityModel()
        with pytest.raises(RuntimeError, match="fit"):
            model.predict_proba(0, 0, 100, 120, 5.0)

    def test_coefficient_table_shape(self, fitted_wp_model):
        df = fitted_wp_model.coefficient_table()
        assert df.shape == (8, 2)
        assert list(df.columns) == ["feature", "coefficient"]


# ─────────────────────────────────────────────────────────────────────────────
# 2. Prediction sanity tests
# ─────────────────────────────────────────────────────────────────────────────

class TestWinProbPrediction:

    def test_output_in_zero_one(self, fitted_wp_model):
        prob = fitted_wp_model.predict_proba(
            over_number=10, wickets_before=2,
            runs_required=60, balls_remaining=60,
            required_run_rate=6.0,
        )
        assert 0.0 <= prob <= 1.0

    def test_comfortable_chase_high_prob(self, fitted_wp_model):
        """
        Needing 10 runs off 60 balls with 0 wickets lost → high win probability.
        """
        prob = fitted_wp_model.predict_proba(
            over_number=10, wickets_before=0,
            runs_required=10, balls_remaining=60,
            required_run_rate=1.0,
        )
        assert prob > 0.80, f"Comfortable state has prob={prob:.3f}, expected >0.80"

    def test_desperate_chase_low_prob(self, fitted_wp_model):
        """
        Needing 80 runs off 12 balls with 8 wickets fallen → low win probability.
        """
        prob = fitted_wp_model.predict_proba(
            over_number=18, wickets_before=8,
            runs_required=80, balls_remaining=12,
            required_run_rate=40.0,
        )
        assert prob < 0.15, f"Desperate state has prob={prob:.3f}, expected <0.15"

    def test_monotone_in_runs_required(self, fitted_wp_model):
        """
        All else equal, higher runs_required → lower win probability.
        """
        base_kwargs = dict(over_number=10, wickets_before=3,
                          balls_remaining=60)
        probs = []
        for rr in [20, 50, 80, 110]:
            rrr = (rr * 6.0) / 60
            p = fitted_wp_model.predict_proba(
                runs_required=rr, required_run_rate=rrr, **base_kwargs
            )
            probs.append(p)
        # Probabilities must be strictly decreasing.
        for i in range(len(probs) - 1):
            assert probs[i] > probs[i + 1], (
                f"Win prob not monotone in runs_required: {probs}"
            )

    def test_monotone_in_wickets(self, fitted_wp_model):
        """
        All else equal, more wickets fallen → lower win probability.
        """
        base_kwargs = dict(over_number=10, runs_required=60,
                           balls_remaining=60, required_run_rate=6.0)
        probs = [
            fitted_wp_model.predict_proba(wickets_before=w, **base_kwargs)
            for w in [0, 2, 4, 7, 9]
        ]
        for i in range(len(probs) - 1):
            assert probs[i] > probs[i + 1], (
                f"Win prob not monotone in wickets: {probs}"
            )

    def test_batch_matches_single(self, fitted_wp_model, train_inn2_with_outcome):
        """batch prediction must match single-row prediction."""
        train, _ = train_inn2_with_outcome
        sample = train.iloc[:10].copy()
        batch_probs = fitted_wp_model.predict_proba_batch(sample)
        for i, row in sample.iterrows():
            single = fitted_wp_model.predict_proba(
                row["over_number"], row["wickets_before"],
                row["runs_required"], row["balls_remaining"],
                row["required_run_rate"],
            )
            idx = list(sample.index).index(i)
            assert abs(batch_probs[idx] - single) < 1e-6, (
                f"Batch vs single mismatch at row {i}: {batch_probs[idx]:.6f} vs {single:.6f}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Evaluation metric tests
# ─────────────────────────────────────────────────────────────────────────────

class TestWinProbEvaluation:

    def test_train_accuracy_above_chance(self, fitted_wp_model, train_inn2_with_outcome):
        """Train accuracy must beat 50% random guessing."""
        train, _ = train_inn2_with_outcome
        metrics = fitted_wp_model.evaluate(train)
        assert metrics["accuracy"] > 0.60, (
            f"Train accuracy {metrics['accuracy']:.3f} should be > 0.60"
        )

    def test_test_accuracy_above_chance(self, fitted_wp_model, train_inn2_with_outcome):
        """Test accuracy must beat 50% random guessing (no overfitting)."""
        _, test = train_inn2_with_outcome
        metrics = fitted_wp_model.evaluate(test)
        assert metrics["accuracy"] > 0.58, (
            f"Test accuracy {metrics['accuracy']:.3f} should be > 0.58"
        )

    def test_brier_score_below_0_25(self, fitted_wp_model, train_inn2_with_outcome):
        """
        Brier score (MSE of probability vs label) should be below 0.25.
        A random model would score 0.25 (predicting 0.5 always).
        """
        train, _ = train_inn2_with_outcome
        metrics = fitted_wp_model.evaluate(train)
        assert metrics["brier_score"] < 0.25, (
            f"Brier score {metrics['brier_score']:.4f} >= 0.25 (random baseline)"
        )

    def test_log_loss_below_0_693(self, fitted_wp_model, train_inn2_with_outcome):
        """Log-loss below ln(2)=0.693 (random baseline)."""
        train, _ = train_inn2_with_outcome
        metrics = fitted_wp_model.evaluate(train)
        assert metrics["log_loss"] < 0.693, (
            f"Log-loss {metrics['log_loss']:.4f} >= 0.693"
        )

    def test_evaluate_keys(self, fitted_wp_model, train_inn2_with_outcome):
        train, _ = train_inn2_with_outcome
        metrics = fitted_wp_model.evaluate(train)
        for key in ["accuracy", "log_loss", "brier_score", "n_samples"]:
            assert key in metrics, f"Missing key '{key}' in evaluate() output"

    def test_no_data_leakage(self, fitted_wp_model, train_inn2_with_outcome):
        """
        Test accuracy should not be significantly below train accuracy.
        Large gap (>15 pp) would indicate overfitting on train set.
        """
        train, test = train_inn2_with_outcome
        train_acc = fitted_wp_model.evaluate(train)["accuracy"]
        test_acc  = fitted_wp_model.evaluate(test)["accuracy"]
        gap = train_acc - test_acc
        assert gap < 0.15, (
            f"Train-test accuracy gap {gap:.3f} suggests overfitting "
            f"(train={train_acc:.3f}, test={test_acc:.3f})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 4. Helper function tests
# ─────────────────────────────────────────────────────────────────────────────

class TestHelpers:

    def test_sigmoid_range(self):
        # Use moderate inputs that don't underflow/overflow at float64.
        z = np.array([-10, -1, 0, 1, 10], dtype=float)
        s = _sigmoid(z)
        assert np.all(s >= 0) and np.all(s <= 1)
        # Middle value must be strictly between 0 and 1.
        assert 0.0 < float(s[2]) < 1.0

    def test_sigmoid_at_zero(self):
        assert abs(_sigmoid(np.array([0.0]))[0] - 0.5) < 1e-10

    def test_feature_matrix_shape(self):
        X = _build_feature_matrix(
            np.array([5.0]), np.array([2.0]), np.array([60.0]),
            np.array([60.0]), np.array([6.0]),
        )
        assert X.shape == (1, 7)

    def test_feature_matrix_rrr_capped(self):
        X = _build_feature_matrix(
            np.array([19.0]), np.array([9.0]), np.array([50.0]),
            np.array([6.0]), np.array([999.0]),   # extreme RRR
        )
        # Column 4 = required_run_rate, capped at 36.
        from win_probability import RRR_CAP
        assert X[0, 4] == RRR_CAP

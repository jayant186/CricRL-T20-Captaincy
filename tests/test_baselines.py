import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from baselines import (
    AlwaysConservative, AlwaysBalanced, AlwaysAggressive,
    RRRPolicy, evaluate_policy, compare_policies,
)
from outcome_model import OutcomeModel
from simulator import T20Simulator

DATA = Path(__file__).resolve().parents[1] / "data" / "processed"



# Fixtures


@pytest.fixture(scope="session")
def fitted_model():
    from data_loader import load_pipeline_data
    data = load_pipeline_data(verbose=False)
    train_ids = set(data["train_match_ids"])
    inn2 = pd.read_csv(DATA / "innings2_chase.csv", low_memory=False)
    train = inn2[inn2["match_id"].isin(train_ids)].copy()
    return OutcomeModel().fit(train)


@pytest.fixture(scope="session")
def simulator(fitted_model):
    return T20Simulator(fitted_model, random_seed=42)



# 1. Policy interface tests


class TestPolicyInterface:

    def _make_running_sim(self, simulator):
        """Return the simulator mid-episode (after reset)."""
        simulator.reset(target=165, random_seed=0)
        return simulator

    def test_always_conservative_returns_0(self, simulator):
        sim = self._make_running_sim(simulator)
        assert AlwaysConservative().select_action(sim) == 0

    def test_always_balanced_returns_1(self, simulator):
        sim = self._make_running_sim(simulator)
        assert AlwaysBalanced().select_action(sim) == 1

    def test_always_aggressive_returns_2(self, simulator):
        sim = self._make_running_sim(simulator)
        assert AlwaysAggressive().select_action(sim) == 2

    def test_all_policies_have_name(self):
        for policy in [AlwaysConservative(), AlwaysBalanced(),
                       AlwaysAggressive(), RRRPolicy()]:
            assert isinstance(policy.name, str) and len(policy.name) > 0

    def test_rrr_policy_action_in_valid_range(self, simulator):
        """RRRPolicy must always return 0, 1, or 2."""
        policy = RRRPolicy()
        simulator.reset(target=165, random_seed=0)
        actions = set()
        for _ in range(20):
            a = policy.select_action(simulator)
            assert a in (0, 1, 2)
            actions.add(a)
            if simulator.is_done:
                break
            simulator.step(aggression=a)



# 2. RRRPolicy logic tests


class TestRRRPolicy:

    class _MockSim:
        """Minimal mock with just the attributes RRRPolicy reads."""
        def __init__(self, score, legal_balls, target):
            self.score = score
            self.legal_balls = legal_balls
            self.target = target

    def test_comfortable_rrr_gives_conservative(self):
        """RRR = 4 runs/over → Conservative."""
        # Need 20 runs off 30 balls → RRR = 20*6/30 = 4.0
        sim = self._MockSim(score=145, legal_balls=90, target=165)
        assert RRRPolicy().select_action(sim) == 0

    def test_moderate_rrr_gives_balanced(self):
        """RRR = 8 runs/over → Balanced."""
        # Need 40 runs off 30 balls → RRR = 40*6/30 = 8.0
        sim = self._MockSim(score=125, legal_balls=90, target=165)
        assert RRRPolicy().select_action(sim) == 1

    def test_high_rrr_gives_aggressive(self):
        """RRR = 15 runs/over → Aggressive."""
        # Need 50 runs off 20 balls → RRR = 50*6/20 = 15.0
        sim = self._MockSim(score=115, legal_balls=100, target=165)
        assert RRRPolicy().select_action(sim) == 2

    def test_custom_thresholds(self):
        """Custom thresholds should be respected."""
        # Need 30 runs off 30 balls → RRR = 6.0.
        sim = self._MockSim(score=135, legal_balls=90, target=165)
        # Default: 6.0 < 7.0 → Conservative.
        assert RRRPolicy(low_thresh=7.0).select_action(sim) == 0
        # Modified: 6.0 >= 5.0 and < 8.0 → Balanced.
        assert RRRPolicy(low_thresh=5.0, high_thresh=8.0).select_action(sim) == 1
        # Modified: 6.0 >= 4.0 and >= 5.0 → Aggressive.
        assert RRRPolicy(low_thresh=4.0, high_thresh=5.0).select_action(sim) == 2


# 3. evaluate_policy() tests


class TestEvaluatePolicy:

    N = 100

    def test_returns_dict_with_required_keys(self, simulator):
        result = evaluate_policy(AlwaysBalanced(), simulator,
                                 n_episodes=self.N, target=165)
        expected = {
            "policy_name", "n_episodes", "target", "win_rate",
            "mean_score", "std_score", "mean_wickets", "mean_legal_balls",
            "all_out_rate", "overs_complete_rate",
        }
        assert expected.issubset(set(result.keys()))

    def test_win_rate_in_range(self, simulator):
        result = evaluate_policy(AlwaysBalanced(), simulator,
                                 n_episodes=self.N, target=165)
        assert 0.0 <= result["win_rate"] <= 1.0

    def test_mean_legal_balls_valid(self, simulator):
        result = evaluate_policy(AlwaysBalanced(), simulator,
                                 n_episodes=self.N, target=165)
        assert 0 <= result["mean_legal_balls"] <= 120

    def test_n_episodes_correct(self, simulator):
        result = evaluate_policy(AlwaysBalanced(), simulator,
                                 n_episodes=self.N, target=165)
        assert result["n_episodes"] == self.N

    def test_policy_name_in_result(self, simulator):
        result = evaluate_policy(AlwaysConservative(), simulator,
                                 n_episodes=self.N, target=165)
        assert result["policy_name"] == "AlwaysConservative"

    def test_all_out_plus_overs_complete_le_one(self, simulator):
        """all_out_rate + overs_complete_rate must be <= 1 (can't both be 100%)."""
        result = evaluate_policy(AlwaysBalanced(), simulator,
                                 n_episodes=self.N, target=165)
        assert result["all_out_rate"] + result["overs_complete_rate"] <= 1.01



# 4. Relative policy ordering tests (large-sample)


class TestPolicyOrdering:
    

    N = 200

    @pytest.fixture(scope="class")
    @classmethod
    def results(cls, simulator):
        r = {}
        for PolicyClass in [AlwaysConservative, AlwaysBalanced, AlwaysAggressive, RRRPolicy]:
            p = PolicyClass()
            r[p.name] = evaluate_policy(p, simulator, cls.N, target=175)
        return r

    def test_aggressive_wins_more_than_conservative_hard_target(self, results):
        """On a hard target (175), aggressive should win more than conservative."""
        agg = results["AlwaysAggressive"]["win_rate"]
        con = results["AlwaysConservative"]["win_rate"]
        # Allow a 3 pp tolerance to handle sampling variance.
        assert agg >= con - 0.03, (
            f"Aggressive win_rate {agg:.3f} < Conservative {con:.3f} - 0.03 "
            "(unexpected: aggressive should outperform on high targets)"
        )

    def test_conservative_fewer_wickets_lost(self, results):
        """Conservative should lose fewer wickets than Aggressive."""
        con_wkt = results["AlwaysConservative"]["mean_wickets"]
        agg_wkt = results["AlwaysAggressive"]["mean_wickets"]
        assert con_wkt <= agg_wkt, (
            f"Conservative mean_wickets {con_wkt:.2f} > Aggressive {agg_wkt:.2f}"
        )

    def test_rrr_win_rate_between_extremes(self, results):
        """RRR policy win rate should be in the ballpark of balanced."""
        rrr_wr  = results["RRRPolicy"]["win_rate"]
        con_wr  = results["AlwaysConservative"]["win_rate"]
        agg_wr  = results["AlwaysAggressive"]["win_rate"]
        lo = min(con_wr, agg_wr) - 0.10
        hi = max(con_wr, agg_wr) + 0.10
        assert lo <= rrr_wr <= hi, (
            f"RRRPolicy win rate {rrr_wr:.3f} outside [{lo:.3f}, {hi:.3f}]"
        )



# 5. compare_policies() tests


class TestComparePolicies:

    def test_returns_dataframe(self, simulator):
        policies = [AlwaysConservative(), AlwaysBalanced(), AlwaysAggressive()]
        df = compare_policies(policies, simulator, n_episodes=50, targets=[160, 180])
        assert isinstance(df, pd.DataFrame)

    def test_row_count(self, simulator):
        policies = [AlwaysConservative(), AlwaysBalanced()]
        df = compare_policies(policies, simulator, n_episodes=30, targets=[155, 165, 175])
        # 2 policies × 3 targets = 6 rows
        assert len(df) == 6

    def test_required_columns(self, simulator):
        policies = [AlwaysBalanced()]
        df = compare_policies(policies, simulator, n_episodes=30, targets=[165])
        for col in ["policy_name", "target", "win_rate", "mean_score"]:
            assert col in df.columns

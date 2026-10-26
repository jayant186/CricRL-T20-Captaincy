""" Unit and Integration Tests for CricketEnv """

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_loader import load_pipeline_data
from outcome_model import OutcomeModel
from rl_env import CricketEnv
from simulator import T20Simulator
from win_probability import WinProbabilityModel

DATA = Path(__file__).resolve().parents[1] / "data" / "processed"


@pytest.fixture(scope="session")
def trained_models():
    """Fit OutcomeModel and WinProbabilityModel once for the test session."""
    data = load_pipeline_data(verbose=False)
    train_ids = set(data["train_match_ids"])

    inn2 = pd.read_csv(DATA / "innings2_chase.csv", low_memory=False)
    ms = pd.read_csv(DATA / "match_summary_clean.csv", low_memory=False)

    ms2 = ms[ms["innings"] == 2][["match_id", "batting_team_canonical", "winner"]].copy()
    ms2["batting_team_won"] = ms2["batting_team_canonical"] == ms2["winner"]
    inn2 = inn2.merge(ms2[["match_id", "batting_team_won"]], on="match_id", how="left")

    train_df = inn2[inn2["match_id"].isin(train_ids)].copy()

    outcome_model = OutcomeModel().fit(train_df)
    wp_model = WinProbabilityModel().fit(train_df)

    return outcome_model, wp_model

# 1. Environment Specification Tests
class TestCricketEnvSpecs:
    def test_action_space_is_discrete_3(self, trained_models):
        om, _ = trained_models
        env = CricketEnv(outcome_model=om)
        assert env.action_space.n == 3
        assert env.action_space.contains(0)
        assert env.action_space.contains(1)
        assert env.action_space.contains(2)
        assert not env.action_space.contains(3)
        assert not env.action_space.contains(-1)

    def test_observation_space_specs(self, trained_models):
        om, _ = trained_models
        env = CricketEnv(outcome_model=om)
        assert env.observation_space.shape == (6,)
        assert env.observation_space.dtype == np.float32
        assert np.all(env.observation_space.low == 0.0)
        assert np.all(env.observation_space.high == 1.0)

    def test_invalid_reward_type_raises(self, trained_models):
        om, _ = trained_models
        with pytest.raises(ValueError, match="reward_type"):
            CricketEnv(outcome_model=om, reward_type="unsupported")

    def test_shaped_without_wp_model_raises(self, trained_models):
        om, _ = trained_models
        with pytest.raises(ValueError, match="win_prob_model"):
            CricketEnv(outcome_model=om, reward_type="shaped", win_prob_model=None)

    def test_missing_simulator_and_outcome_model_raises(self):
        with pytest.raises(ValueError, match="either simulator or outcome_model"):
            CricketEnv(simulator=None, outcome_model=None)

# 2. Reset Behavior Tests
class TestCricketEnvReset:
    def test_reset_return_signature(self, trained_models):
        om, _ = trained_models
        env = CricketEnv(outcome_model=om)
        obs, info = env.reset(seed=42)

        assert isinstance(obs, np.ndarray)
        assert obs.shape == (6,)
        assert obs.dtype == np.float32
        assert env.observation_space.contains(obs)
        assert isinstance(info, dict)
        assert "score" in info
        assert "wickets" in info
        assert "target" in info

    def test_reset_with_options_target(self, trained_models):
        om, _ = trained_models
        env = CricketEnv(outcome_model=om)
        _, info = env.reset(options={"target": 182})
        assert info["target"] == 182
        assert env.simulator.target == 182

    def test_reset_with_tuple_target_distribution(self, trained_models):
        om, _ = trained_models
        env = CricketEnv(outcome_model=om, target_distribution=(140, 160))
        targets = []
        for s in range(20):
            _, info = env.reset(seed=s)
            targets.append(info["target"])
            assert 140 <= info["target"] <= 160
        assert len(set(targets)) > 1  # Verify variance

    def test_reset_with_list_target_distribution(self, trained_models):
        om, _ = trained_models
        allowed = [145, 165, 185]
        env = CricketEnv(outcome_model=om, target_distribution=allowed)
        for s in range(15):
            _, info = env.reset(seed=s)
            assert info["target"] in allowed

# 3. Step and Rewards 
class TestCricketEnvStep:
    def test_step_return_signature(self, trained_models):
        om, _ = trained_models
        env = CricketEnv(outcome_model=om)
        env.reset(seed=42)
        obs, reward, terminated, truncated, info = env.step(1)

        assert isinstance(obs, np.ndarray)
        assert obs.shape == (6,)
        assert obs.dtype == np.float32
        assert env.observation_space.contains(obs)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert not truncated
        assert isinstance(info, dict)

    def test_step_invalid_action_raises(self, trained_models):
        om, _ = trained_models
        env = CricketEnv(outcome_model=om)
        env.reset(seed=42)
        with pytest.raises(ValueError):
            env.step(5)

    def test_sparse_reward_values(self, trained_models):
        om, _ = trained_models
        env = CricketEnv(outcome_model=om, reward_type="sparse")
        env.reset(seed=42, options={"target": 165})

        terminated = False
        step_rewards = []
        while not terminated:
            _, r, terminated, _, info = env.step(1)
            step_rewards.append(r)

        # All intermediate rewards must be 0.0
        for r in step_rewards[:-1]:
            assert r == 0.0

        # Terminal reward must be +1.0 (win) or -1.0 (loss)
        final_reward = step_rewards[-1]
        if info["won"]:
            assert final_reward == 1.0
        else:
            assert final_reward == -1.0

    def test_shaped_reward_formula(self, trained_models):
        """
        Verify corrected potential-based reward shaping per-step (Ng et al. 1999):
          r_t = R(s,a,s') + gamma * Phi(s') - Phi(s)
        where R is sparse (0 non-terminal, +1/-1 terminal).

        We verify the identity by comparing the shaped reward returned by env.step()
        against the manually computed F(s,a,s') = gamma * Phi(s') - Phi(s) + R.
        """
        om, wp_model = trained_models
        gamma = 0.99
        env = CricketEnv(
            outcome_model=om,
            win_prob_model=wp_model,
            reward_type="shaped",
            gamma=gamma,
        )
        env.reset(seed=123, options={"target": 165})

        # Step through a few deliveries and verify the per-step formula
        mismatches = 0
        for _ in range(30):
            if env.simulator.is_done:
                break
            phi_s = env._current_wp   # Phi(s) before step
            _, r_env, terminated, _, info = env.step(1)
            phi_s_prime = env._current_wp  # Phi(s') after step (already updated)

            # Sparse reward on this step
            if terminated:
                R = 1.0 if info["won"] else -1.0
            else:
                R = 0.0

            expected_r = R + gamma * phi_s_prime - phi_s
            if abs(r_env - expected_r) > 1e-5:
                mismatches += 1

        assert mismatches == 0, (
            f"Shaped reward F(s,a,s') = R + gamma*Phi(s') - Phi(s) failed "
            f"on {mismatches} step(s)"
        )

    def test_gamma_stored_on_env(self, trained_models):
        """CricketEnv must expose a gamma attribute matching the constructor arg."""
        om, wp_model = trained_models
        env = CricketEnv(outcome_model=om, win_prob_model=wp_model,
                         reward_type="shaped", gamma=0.95)
        assert env.gamma == 0.95


    def test_reproducibility_with_seed(self, trained_models):
        om, _ = trained_models
        env1 = CricketEnv(outcome_model=om)
        env2 = CricketEnv(outcome_model=om)

        obs1, _ = env1.reset(seed=999, options={"target": 165})
        obs2, _ = env2.reset(seed=999, options={"target": 165})
        np.testing.assert_array_equal(obs1, obs2)

        for action in [1, 2, 0, 1, 2, 0]:
            o1, r1, d1, _, _ = env1.step(action)
            o2, r2, d2, _, _ = env2.step(action)
            np.testing.assert_array_equal(o1, o2)
            assert r1 == r2
            assert d1 == d2
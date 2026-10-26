"""
tests/test_interactive_app.py — Unit Tests for Interactive Chase Demo Logic
=============================================================================

Tests:
1. Model loading and Q-value generation for interactive stepping
2. Step-by-step match state transitions
3. Full innings chase simulation to completion
4. CSV decision log structure and export validation
"""

import sys
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import pytest

# Ensure src/ is on sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_loader import load_pipeline_data
from dqn_agent import DQNAgent
from outcome_model import OutcomeModel
from simulator import T20Simulator


@pytest.fixture(scope="module")
def loaded_models():
    """Load fitted outcome model and DQN agent."""
    data = load_pipeline_data(verbose=False)
    train_ids = set(data["train_match_ids"])
    inn2 = pd.read_csv(ROOT / "data" / "processed" / "innings2_chase.csv", low_memory=False)
    train_df = inn2[inn2["match_id"].isin(train_ids)].copy()
    outcome_model = OutcomeModel().fit(train_df)

    agent = DQNAgent(state_dim=6, action_dim=3)
    ckpt_path = ROOT / "models" / "dqn_cricket.pt"
    if ckpt_path.exists():
        agent.load(ckpt_path)
    return outcome_model, agent


class TestInteractiveChaseDemo:
    def test_q_values_and_action_selection(self, loaded_models):
        outcome_model, agent = loaded_models
        sim = T20Simulator(outcome_model=outcome_model, random_seed=42)
        obs = sim.reset(target=165, initial_wickets=0, random_seed=42)

        q_vals = agent.get_q_values(obs)
        assert isinstance(q_vals, np.ndarray)
        assert q_vals.shape == (3,)
        assert np.isfinite(q_vals).all()

        action = agent.select_action(obs, eval_mode=True)
        assert action == int(np.argmax(q_vals))
        assert action in (0, 1, 2)

    def test_single_delivery_simulation_step(self, loaded_models):
        outcome_model, agent = loaded_models
        sim = T20Simulator(outcome_model=outcome_model, random_seed=42)
        obs = sim.reset(target=165, initial_wickets=0, random_seed=42)

        q_vals = agent.get_q_values(obs)
        action = agent.select_action(obs, eval_mode=True)

        score_before = sim.score
        wickets_before = sim.wickets

        next_obs, reward, done, info = sim.step(action)
        outcome = sim.last_outcome

        assert outcome is not None
        assert 0 <= outcome.batter_runs <= 6
        assert sim.score == score_before + outcome.total_runs
        assert sim.wickets == wickets_before + (1 if outcome.is_wicket else 0)

    def test_complete_chase_simulation(self, loaded_models):
        outcome_model, agent = loaded_models
        sim = T20Simulator(outcome_model=outcome_model, random_seed=123)
        sim.reset(target=165, initial_wickets=0, random_seed=123)

        records = []
        while not sim.is_done:
            obs = sim._observation()
            q_vals = agent.get_q_values(obs)
            action = agent.select_action(obs, eval_mode=True)

            sim.step(action)
            outcome = sim.last_outcome

            records.append({
                "ball_num": len(records) + 1,
                "q_conservative": float(q_vals[0]),
                "q_balanced": float(q_vals[1]),
                "q_aggressive": float(q_vals[2]),
                "action": action,
                "runs": outcome.total_runs,
                "is_wicket": outcome.is_wicket,
                "score": sim.score,
                "wickets": sim.wickets,
            })

        assert sim.is_done
        assert isinstance(sim.won, bool)
        assert sim.legal_balls <= 120
        assert sim.wickets <= 10
        if sim.won:
            assert sim.score >= 165
        else:
            assert sim.score < 165 or sim.wickets == 10

        # Verify DataFrame and CSV export
        df = pd.DataFrame(records)
        assert len(df) == len(records)
        assert "q_conservative" in df.columns
        assert "q_balanced" in df.columns
        assert "q_aggressive" in df.columns

        with tempfile.TemporaryDirectory() as tmp_dir:
            test_csv = Path(tmp_dir) / "test_chase_log.csv"
            df.to_csv(test_csv, index=False)
            assert test_csv.exists()
            loaded_df = pd.read_csv(test_csv)
            assert len(loaded_df) == len(df)

    def test_integrated_bowling_chase_simulation(self, loaded_models):
        from bowling_policy import get_csk_squad, PhaseSpecialistCaptaincy
        outcome_model, agent = loaded_models

        squad = get_csk_squad()
        captaincy = PhaseSpecialistCaptaincy()
        sim = T20Simulator(outcome_model=outcome_model, random_seed=42)
        sim.reset(target=165, initial_wickets=0, random_seed=42)

        active_b, _ = captaincy.select_bowler(squad, 0, 0, 0, 165)
        active_t, _ = captaincy.select_tactic(active_b, 0, 0, 0, 165)

        while not sim.is_done:
            obs = sim._observation()
            batting_action = agent.select_action(obs, eval_mode=True)

            sim.step(
                aggression=batting_action,
                bowling_aggression=active_t,
                bowler=active_b,
            )
            outcome = sim.last_outcome

            if outcome.is_legal and sim.legal_balls % 6 == 0 and not sim.is_done:
                squad.complete_over(active_b.name)
                active_b, _ = captaincy.select_bowler(squad, sim.legal_balls, sim.score, sim.wickets, sim.target)
                active_t, _ = captaincy.select_tactic(active_b, sim.legal_balls, sim.score, sim.wickets, sim.target)

        assert sim.is_done
        for b in squad.bowlers:
            assert b.legal_balls <= 24
        summary = squad.get_figures_summary()
        assert not summary.empty
        assert "Bowler" in summary.columns

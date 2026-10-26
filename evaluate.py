

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

# Ensure src/ is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from baselines import (
    AlwaysAggressive,
    AlwaysBalanced,
    AlwaysConservative,
    RRRPolicy,
)
from data_loader import load_pipeline_data
from dqn_agent import DQNAgent
from outcome_model import OutcomeModel
from simulator import T20Simulator

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = ROOT / "models" / "dqn_cricket.pt"
DEFAULT_OUTPUT_CSV = ROOT / "reports" / "evaluation_results.csv"


def wilson_ci(wins: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """
    Compute Wilson score interval for binomial proportion.
    """
    if n == 0:
        return 0.0, 0.0
    z = 1.96  # 95% confidence
    p = wins / n
    denominator = 1.0 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denominator
    margin = (z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


class DQNAgentPolicyWrapper:
    """
    Wraps DQNAgent to conform to the BaselinePolicy interface:
        policy.select_action(simulator) -> int
    """

    def __init__(self, agent: DQNAgent):
        self.agent = agent
        self.name = "DQNAgent (Ours)"

    def select_action(self, simulator: T20Simulator) -> int:
        obs = simulator._observation()
        return self.agent.select_action(obs, eval_mode=True)


def evaluate_policy_on_target(
    policy,
    simulator: T20Simulator,
    target: int,
    n_episodes: int = 500,
    base_seed: int = 2026,
) -> dict:
    """
    Evaluate a policy on a fixed target over n_episodes with deterministic seeds.
    """
    scores = []
    wickets = []
    legal_balls = []
    wins = 0
    all_outs = 0
    overs_completes = 0

    for ep in range(n_episodes):
        ep_seed = base_seed + ep
        obs = simulator.reset(target=target, initial_wickets=0, random_seed=ep_seed)

        while not simulator.is_done:
            if hasattr(policy, "select_action"):
                action = policy.select_action(simulator)
            else:
                action = policy(simulator)
            simulator.step(action)

        scores.append(simulator.score)
        wickets.append(simulator.wickets)
        legal_balls.append(simulator.legal_balls)

        if simulator.won:
            wins += 1
        elif simulator.wickets == 10:
            all_outs += 1
        elif simulator.legal_balls == 120:
            overs_completes += 1

    win_rate = wins / n_episodes
    ci_low, ci_high = wilson_ci(wins, n_episodes)

    return {
        "policy": getattr(policy, "name", str(policy)),
        "target": target,
        "n_episodes": n_episodes,
        "win_rate": round(win_rate, 4),
        "ci_95_low": round(ci_low, 4),
        "ci_95_high": round(ci_high, 4),
        "mean_score": round(float(np.mean(scores)), 2),
        "std_score": round(float(np.std(scores)), 2),
        "mean_wickets": round(float(np.mean(wickets)), 2),
        "mean_legal_balls": round(float(np.mean(legal_balls)), 2),
        "all_out_rate": round(all_outs / n_episodes, 4),
        "overs_complete_rate": round(overs_completes / n_episodes, 4),
    }


def run_full_evaluation(
    model_path: Path = DEFAULT_MODEL_PATH,
    targets: list[int] = [145, 165, 185],
    n_episodes: int = 500,
    output_csv: Path = DEFAULT_OUTPUT_CSV,
    seed: int = 2026,
) -> pd.DataFrame:
    """
    Run complete benchmark comparing all baselines and DQN agent.
    """
    print(f"Loading OutcomeModel fitted strictly on training partition...")
    data = load_pipeline_data(verbose=False)
    train_ids = set(data["train_match_ids"])

    inn2 = pd.read_csv(ROOT / "data" / "processed" / "innings2_chase.csv", low_memory=False)
    train_df = inn2[inn2["match_id"].isin(train_ids)].copy()
    outcome_model = OutcomeModel().fit(train_df)

    sim = T20Simulator(outcome_model=outcome_model, random_seed=seed)

    # Initialize DQN agent and load trained weights
    agent = DQNAgent(state_dim=6, action_dim=3)
    model_path = Path(model_path)
    if model_path.exists():
        print(f"Loading trained DQN agent from: {model_path}")
        agent.load(model_path)
        dqn_policy = DQNAgentPolicyWrapper(agent)
    else:
        print(f"WARNING: Model file {model_path} not found. Running only heuristic baselines.")
        dqn_policy = None

    policies = [
        AlwaysConservative(),
        AlwaysBalanced(),
        AlwaysAggressive(),
        RRRPolicy(),
    ]
    if dqn_policy is not None:
        policies.append(dqn_policy)

    records = []
    print("\n" + "=" * 80)
    print(f"Running Benchmark ({len(policies)} policies x {len(targets)} targets x {n_episodes} episodes)")
    print("=" * 80)

    for target in targets:
        print(f"\n--- Target: {target} runs ---")
        for policy in policies:
            res = evaluate_policy_on_target(
                policy=policy,
                simulator=sim,
                target=target,
                n_episodes=n_episodes,
                base_seed=seed,
            )
            records.append(res)
            print(
                f"  {res['policy']:<20} | "
                f"Win: {res['win_rate']*100:5.1f}% [{res['ci_95_low']*100:4.1f}% - {res['ci_95_high']*100:4.1f}%] | "
                f"Score: {res['mean_score']:5.1f} | "
                f"Wkts: {res['mean_wickets']:4.1f} | "
                f"All-Out: {res['all_out_rate']*100:4.1f}%"
            )

    df_results = pd.DataFrame(records)

    # Save CSV
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(output_csv, index=False)
    print(f"\nSaved evaluation benchmark to: {output_csv}")

    return df_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CricRL DQN Agent vs Baselines")
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--n-episodes", type=int, default=500)
    parser.add_argument("--targets", nargs="+", type=int, default=[145, 165, 185])
    parser.add_argument("--output-csv", type=str, default=str(DEFAULT_OUTPUT_CSV))
    parser.add_argument("--seed", type=int, default=2026)

    args = parser.parse_args()

    run_full_evaluation(
        model_path=Path(args.model_path),
        targets=args.targets,
        n_episodes=args.n_episodes,
        output_csv=Path(args.output_csv),
        seed=args.seed,
    )

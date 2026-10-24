

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch

# Ensure src/ is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from baselines import AlwaysBalanced, RRRPolicy
from data_loader import load_pipeline_data
from dqn_agent import DQNAgent
from outcome_model import OutcomeModel
from rl_env import CricketEnv
from simulator import T20Simulator
from win_probability import WinProbabilityModel

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "processed"
DEFAULT_MODEL_PATH = ROOT / "models" / "dqn_cricket.pt"
DEFAULT_LOG_PATH = ROOT / "reports" / "training_log.csv"


def prepare_training_data() -> tuple[pd.DataFrame, OutcomeModel, WinProbabilityModel]:
    """
    Load data using data_loader split and fit OutcomeModel and WinProbabilityModel
    strictly on training-split matches.
    """
    print("[1/4] Loading pipeline data and strictly isolating training matches...")
    data = load_pipeline_data(verbose=False)
    train_ids = set(data["train_match_ids"])

    inn2 = pd.read_csv(DATA_DIR / "innings2_chase.csv", low_memory=False)
    ms = pd.read_csv(DATA_DIR / "match_summary_clean.csv", low_memory=False)

    # Attach outcome label
    ms2 = ms[ms["innings"] == 2][["match_id", "batting_team_canonical", "winner"]].copy()
    ms2["batting_team_won"] = ms2["batting_team_canonical"] == ms2["winner"]
    inn2 = inn2.merge(ms2[["match_id", "batting_team_won"]], on="match_id", how="left")

    train_df = inn2[inn2["match_id"].isin(train_ids)].copy()
    print(f"      Fitted on {len(train_df):,} deliveries from {len(train_ids)} matches.")

    print("[2/4] Fitting OutcomeModel...")
    outcome_model = OutcomeModel().fit(train_df)

    print("[3/4] Fitting WinProbabilityModel...")
    wp_model = WinProbabilityModel().fit(train_df)

    return train_df, outcome_model, wp_model


def evaluate_agent_deterministic(
    agent: DQNAgent,
    outcome_model: OutcomeModel,
    targets: list[int] = [145, 165, 185],
    n_episodes: int = 100,
    seed: int = 1000,
) -> dict[int, float]:
    """
    Evaluate agent in strictly greedy evaluation mode on fixed targets.
    """
    sim = T20Simulator(outcome_model, random_seed=seed)
    win_rates = {}

    for target in targets:
        wins = 0
        for ep in range(n_episodes):
            obs = sim.reset(target=target, initial_wickets=0, random_seed=seed + ep)
            while not sim.is_done:
                action = agent.select_action(obs, eval_mode=True)
                obs, _, _, _ = sim.step(action)
            if sim.won:
                wins += 1
        win_rates[target] = wins / n_episodes

    return win_rates


def train(
    episodes: int = 2000,
    reward_type: str = "shaped",
    lr: float = 1e-3,
    gamma: float = 0.99,
    batch_size: int = 64,
    eval_freq: int = 500,
    save_path: Path = DEFAULT_MODEL_PATH,
    log_path: Path = DEFAULT_LOG_PATH,
    random_seed: int = 42,
) -> tuple[DQNAgent, pd.DataFrame]:
    """
    Main DQN training loop.
    """
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    _, outcome_model, wp_model = prepare_training_data()

    print("[4/4] Initializing CricketEnv and DQNAgent...")
    env = CricketEnv(
        outcome_model=outcome_model,
        win_prob_model=wp_model,
        reward_type=reward_type,
        target_distribution=(135, 205),
        default_target=165,
        random_seed=random_seed,
        gamma=gamma,  # must match DQNAgent gamma for Ng et al. policy-invariance
    )

    agent = DQNAgent(
        state_dim=6,
        action_dim=3,
        lr=lr,
        gamma=gamma,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=max(10_000, episodes * 40),
        batch_size=batch_size,
        buffer_capacity=100_000,
    )

    print("\n" + "=" * 70)
    print(f"Starting DQN Training: {episodes} episodes | Reward: {reward_type} | Device: {agent.device}")
    print("=" * 70)

    log_records = []
    recent_wins = []
    action_counts = {0: 0, 1: 0, 2: 0}
    best_eval_win = -1.0
    start_time = time.time()

    for ep in range(1, episodes + 1):
        obs, info = env.reset(seed=random_seed + ep)
        ep_reward = 0.0
        ep_losses = []
        ep_balls = 0

        terminated = False
        while not terminated:
            action = agent.select_action(obs, eval_mode=False)
            action_counts[action] += 1
            next_obs, reward, terminated, _, step_info = env.step(action)

            agent.store_transition(obs, action, reward, next_obs, terminated)
            if ep_balls % 2 == 0:
                loss = agent.update()
                if loss is not None:
                    ep_losses.append(loss)

            obs = next_obs
            ep_reward += reward
            ep_balls += 1

        won = bool(step_info["won"])
        recent_wins.append(1 if won else 0)
        if len(recent_wins) > 100:
            recent_wins.pop(0)

        rolling_win_rate = float(np.mean(recent_wins))
        avg_loss = float(np.mean(ep_losses)) if ep_losses else 0.0

        log_records.append({
            "episode": ep,
            "target": step_info["target"],
            "score": step_info["score"],
            "wickets": step_info["wickets"],
            "legal_balls": step_info["legal_balls"],
            "won": int(won),
            "reward": round(ep_reward, 4),
            "rolling_win_rate": round(rolling_win_rate, 4),
            "epsilon": round(agent.epsilon, 4),
            "loss": round(avg_loss, 4),
        })

        if ep % 100 == 0 or ep == episodes:
            elapsed = time.time() - start_time
            print(
                f"Ep {ep:4d}/{episodes} | "
                f"WinRate(100): {rolling_win_rate * 100:5.1f}% | "
                f"Reward: {ep_reward:+6.2f} | "
                f"Eps: {agent.epsilon:.3f} | "
                f"Loss: {avg_loss:6.4f} | "
                f"Time: {elapsed:5.1f}s"
            )

        if ep % eval_freq == 0 or ep == episodes:
            eval_results = evaluate_agent_deterministic(agent, outcome_model, targets=[145, 165, 185], n_episodes=50)
            avg_eval_win = float(np.mean(list(eval_results.values())))
            print(
                f"   >> Periodic Eval @ Ep {ep}: "
                f"T145={eval_results[145]*100:.1f}%, "
                f"T165={eval_results[165]*100:.1f}%, "
                f"T185={eval_results[185]*100:.1f}% | "
                f"Avg={avg_eval_win*100:.1f}%"
            )
            if avg_eval_win > best_eval_win:
                best_eval_win = avg_eval_win
                best_save_path = save_path.with_name(f"{save_path.stem}_best.pt")
                agent.save(best_save_path)
                agent.save(save_path)
                print(f"      [*] New best model saved to: {best_save_path}")

    # Save final model and logs
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    best_save_path = save_path.with_name(f"{save_path.stem}_best.pt")
    if not best_save_path.exists():
        agent.save(best_save_path)
        agent.save(save_path)
    print(f"\nModel checkpoint saved to: {save_path} and {best_save_path}")

    log_df = pd.DataFrame(log_records)
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_df.to_csv(log_path, index=False)
    print(f"Training log saved to: {log_path}")

    # Summary report
    tot_actions = sum(action_counts.values())
    print("\nAction Distribution During Training:")
    print(f"  Conservative (0): {action_counts[0]:,} ({action_counts[0]/tot_actions*100:.1f}%)")
    print(f"  Balanced     (1): {action_counts[1]:,} ({action_counts[1]/tot_actions*100:.1f}%)")
    print(f"  Aggressive   (2): {action_counts[2]:,} ({action_counts[2]/tot_actions*100:.1f}%)")

    return agent, log_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN Captaincy Agent for CricRL")
    parser.add_argument("--episodes", type=int, default=2000, help="Number of training episodes")
    parser.add_argument("--reward", type=str, default="shaped", choices=["sparse", "shaped"], help="Reward type")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--eval-freq", type=int, default=500, help="Evaluation frequency (episodes)")
    parser.add_argument("--save-path", type=str, default=str(DEFAULT_MODEL_PATH), help="Model save path")
    parser.add_argument("--log-path", type=str, default=str(DEFAULT_LOG_PATH), help="Log save path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    train(
        episodes=args.episodes,
        reward_type=args.reward,
        lr=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        eval_freq=args.eval_freq,
        save_path=Path(args.save_path),
        log_path=Path(args.log_path),
        random_seed=args.seed,
    )

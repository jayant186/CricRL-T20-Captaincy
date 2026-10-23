

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Set modern aesthetics
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 11


def plot_training_curves(
    log_csv: Union[str, Path],
    output_png: Union[str, Path] = "reports/training_curves.png",
) -> Path:
    
    log_csv = Path(log_csv)
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(log_csv)

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    # 1. Rolling Win Rate
    ax1 = axes[0]
    ax1.plot(df["episode"], df["rolling_win_rate"] * 100, color="#1f77b4", linewidth=2.0)
    ax1.axhline(50, color="gray", linestyle="--", alpha=0.6, label="50% Par Benchmark")
    ax1.set_ylabel("Win Rate (%) [100-ep window]")
    ax1.set_title("CricRL DQN Training Progress", fontsize=14, fontweight="bold")
    ax1.legend(loc="lower right")
    ax1.set_ylim(0, 100)

    # 2. Episode Reward
    ax2 = axes[1]
    # Exponential moving average of reward
    smoothed_reward = df["reward"].ewm(span=50).mean()
    ax2.plot(df["episode"], df["reward"], color="#aec7e8", alpha=0.3, label="Raw Return")
    ax2.plot(df["episode"], smoothed_reward, color="#2ca02c", linewidth=2.0, label="Smoothed Return (EMA-50)")
    ax2.set_ylabel("Episode Return")
    ax2.legend(loc="upper left")

    # 3. Training Loss
    ax3 = axes[2]
    smoothed_loss = df["loss"].ewm(span=50).mean()
    ax3.plot(df["episode"], df["loss"], color="#ffbb78", alpha=0.3, label="Raw Loss")
    ax3.plot(df["episode"], smoothed_loss, color="#d62728", linewidth=2.0, label="Smoothed Huber Loss (EMA-50)")
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Bellman Loss")
    ax3.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(output_png, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Training curves saved to: {output_png}")
    return output_png


def plot_policy_comparison(
    eval_csv: Union[str, Path],
    output_png: Union[str, Path] = "reports/policy_comparison.png",
) -> Path:
    
    eval_csv = Path(eval_csv)
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(eval_csv)

    targets = sorted(df["target"].unique())
    policies = list(df["policy"].unique())

    # Color palette
    colors = {
        "AlwaysConservative": "#7f7f7f",
        "AlwaysBalanced": "#1f77b4",
        "AlwaysAggressive": "#ff7f0e",
        "RRRPolicy": "#2ca02c",
        "DQNAgent (Ours)": "#d62728",
    }

    fig, ax = plt.subplots(figsize=(11, 6))

    bar_width = 0.15
    indices = np.arange(len(targets))

    for i, policy_name in enumerate(policies):
        p_data = df[df["policy"] == policy_name].sort_values("target")
        win_rates = p_data["win_rate"].values * 100
        y_err_low = (p_data["win_rate"] - p_data["ci_95_low"]).values * 100
        y_err_high = (p_data["ci_95_high"] - p_data["win_rate"]).values * 100
        y_err = [y_err_low, y_err_high]

        x_pos = indices + (i - (len(policies) - 1) / 2) * bar_width
        color = colors.get(policy_name, "#9467bd")

        bars = ax.bar(
            x_pos,
            win_rates,
            bar_width,
            yerr=y_err,
            capsize=4,
            label=policy_name,
            color=color,
            alpha=0.9,
            edgecolor="black",
            linewidth=0.7,
        )

        # Value labels above bars
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=45,
            )

    ax.set_ylabel("Win Rate (%)", fontsize=12, fontweight="bold")
    ax.set_title("CricRL Tactical Captaincy: DQN Agent vs Heuristic Baselines", fontsize=14, fontweight="bold")
    ax.set_xticks(indices)
    ax.set_xticklabels([f"Target {t} runs\n(Easy)" if t == 145 else f"Target {t} runs\n(Par)" if t == 165 else f"Target {t} runs\n(Demanding)" for t in targets], fontsize=11)
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right", frameon=True, fontsize=10)

    plt.tight_layout()
    plt.savefig(output_png, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Policy comparison plot saved to: {output_png}")
    return output_png

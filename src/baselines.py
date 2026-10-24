"""
Purpose -> Provide simple, interpretable baseline policies for comparison with the DQN agent. A well-designed DQN should outperform all baselines; if it doesn't, there is a bug in the training pipeline.
Baselines implemented
1. AlwaysConservative  — always selects aggression=0 on every ball
2. AlwaysBalanced      — always selects aggression=1 on every ball (historical average)
3. AlwaysAggressive    — always selects aggression=2 on every ball
4. RRRPolicy           — selects aggression based on the required run rate (RRR):
       RRR < 7          → Conservative (0)   — comfortable, protect wickets
       7 <= RRR < 10    → Balanced     (1)   — on track
       RRR >= 10        → Aggressive   (2)   — need to accelerate
The RRR thresholds are standard T20 cricket heuristics:
- RRR < 7 runs/over is below the median T20 scoring rate (~7-8 runs/over) → safe.
- RRR >= 10 is above the average achievable rate → must take risks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from simulator import T20Simulator

import numpy as np
import pandas as pd

# Base class
class BaselinePolicy(ABC):
    """Abstract base class for all heuristic baseline policies."""

    @abstractmethod
    def select_action(self, simulator: "T20Simulator") -> int:
        """
        Select an aggression action given the current simulator state
        """

    @property
    def name(self) -> str:
        return self.__class__.__name__

# Fixed-action baselines
class AlwaysConservative(BaselinePolicy):
    """Always play conservatively (aggression=0). Prioritise wicket preservation"""
    def select_action(self, simulator: "T20Simulator") -> int:
        return 0


class AlwaysBalanced(BaselinePolicy):
    """Always play balanced (aggression=1). Historical average strategy"""
    def select_action(self, simulator: "T20Simulator") -> int:
        return 1


class AlwaysAggressive(BaselinePolicy):
    """Always play aggressively (aggression=2). Maximise boundary rate."""
    def select_action(self, simulator: "T20Simulator") -> int:
        return 2

# RRR-adaptive policy
class RRRPolicy(BaselinePolicy):
    """
    Adaptive policy that selects aggression based on the current required run rate
    Thresholds (runs per over):
        RRR<low_thresh  → Conservative
        RRR<high_thresh → Balanced 
        RRR >= high_thresh → Aggressive
    Default thresholds (7 and 10) are standard T20 heuristics.
    """

    def __init__(self, low_thresh: float = 7.0, high_thresh: float = 10.0):
        self.low_thresh  = low_thresh
        self.high_thresh = high_thresh

    def select_action(self, simulator: "T20Simulator") -> int:
        # Edge cases: compute RRR from simulator state directly.
        balls_left = 120 - simulator.legal_balls
        runs_req   = simulator.target - simulator.score

        if balls_left <= 0 or runs_req <= 0:
            # Innings effectively over or target already passed; return Balanced.
            return 1

        rrr = (runs_req * 6.0) / balls_left   # runs per over

        if rrr < self.low_thresh:
            return 0  # Conservative: we're ahead, protect wickets
        elif rrr < self.high_thresh:
            return 1  # Balanced: on track
        else:
            return 2  # Aggressive: need to accelerate

# Evaluation function
def evaluate_policy(
    policy: BaselinePolicy,
    simulator: "T20Simulator",
    n_episodes: int,
    target: int,
    random_seed_start: int = 0,
) -> dict:
    """Run n_episodes with the given policy and return aggregate statistics"""
    wins, scores, wickets_list, lb_list = [], [], [], []

    for seed in range(random_seed_start, random_seed_start + n_episodes):
        simulator.reset(target=target, random_seed=seed)
        while not simulator.is_done:
            action = policy.select_action(simulator)
            simulator.step(action)

        wins.append(simulator.won)
        scores.append(simulator.score)
        wickets_list.append(simulator.wickets)
        lb_list.append(simulator.legal_balls)

    scores_arr  = np.array(scores, dtype=float)
    wkts_arr    = np.array(wickets_list, dtype=float)
    lb_arr      = np.array(lb_list, dtype=float)

    return {
        "policy_name":        policy.name,
        "n_episodes":         n_episodes,
        "target":             target,
        "win_rate":           round(float(np.mean(wins)), 4),
        "mean_score":         round(float(scores_arr.mean()), 2),
        "std_score":          round(float(scores_arr.std()), 2),
        "mean_wickets":       round(float(wkts_arr.mean()), 2),
        "mean_legal_balls":   round(float(lb_arr.mean()), 2),
        "all_out_rate":       round(float((wkts_arr == 10).mean()), 4),
        "overs_complete_rate":round(float((lb_arr == 120).mean()), 4),
    }


def compare_policies(
    policies: list[BaselinePolicy],
    simulator: "T20Simulator",
    n_episodes: int = 500,
    targets: list[int] = (145, 165, 185),
) -> pd.DataFrame:
    """ Run all policies on all targets and return a comparison DataFrame """
    rows = []
    for target in targets:
        for policy in policies:
            r = evaluate_policy(policy, simulator, n_episodes, target)
            rows.append(r)
    df = pd.DataFrame(rows)
    return df.sort_values(["target", "win_rate"], ascending=[True, False]).reset_index(drop=True)
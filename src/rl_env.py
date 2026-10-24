"""
Purpose
Wrap the ball-by-ball T20 cricket chase simulator (`T20Simulator`) into a standard `gymnasium.Env` interface. This enables seamless training of reinforcement learning agents (such as DQN, PPO, or SB3 baselines) to make tactical batting aggression decisions during a run chase.
"""

from __future__ import annotations

from typing import Any, Optional, Union

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from match_state import MatchState
from outcome_model import OutcomeModel
from simulator import MAX_LEGAL_BALLS, MAX_WICKETS, T20Simulator
from win_probability import WinProbabilityModel


class CricketEnv(gym.Env):
    """ Gymnasium environment for a T20 second-innings cricket run chase """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        simulator: Optional[T20Simulator] = None,
        outcome_model: Optional[OutcomeModel] = None,
        win_prob_model: Optional[WinProbabilityModel] = None,
        reward_type: str = "sparse",
        target_distribution: Optional[Union[tuple[int, int], list[int]]] = (130, 210),
        default_target: int = 165,
        random_seed: Optional[int] = 42,
        gamma: float = 0.99,
    ):
        """ Initialize the CricketEnv """
        super().__init__()

        if reward_type not in ("sparse", "shaped"):
            raise ValueError(
                f"reward_type must be 'sparse' or 'shaped', got '{reward_type}'"
            )

        if reward_type == "shaped" and win_prob_model is None:
            raise ValueError("win_prob_model must be provided when reward_type='shaped'")

        self.reward_type = reward_type
        self.win_prob_model = win_prob_model
        self.target_distribution = target_distribution
        self.default_target = default_target
        self._initial_seed = random_seed
        # Discount factor must match the RL optimizer's gamma (Ng et al. 1999).
        self.gamma = float(gamma)

        # Setup simulator
        if simulator is not None:
            self.simulator = simulator
        elif outcome_model is not None:
            self.simulator = T20Simulator(outcome_model, random_seed=random_seed)
        else:
            raise ValueError("Must provide either simulator or outcome_model to CricketEnv")

        # Action space: 0 = Conservative, 1 = Balanced, 2 = Aggressive
        self.action_space = spaces.Discrete(3)

        # Observation space: 6 continuous features in [0.0, 1.0]
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(6,),
            dtype=np.float32,
        )

        self._current_wp: float = 0.5
        self._rng: np.random.Generator = np.random.default_rng(seed=random_seed)

    # Gymnasium API
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """ Reset the environment to start a new chase episode """
        super().reset(seed=seed)

        if seed is not None:
            self._rng = np.random.default_rng(seed=seed)

        # Determine target for this chase
        target: int = self.default_target
        if options is not None and "target" in options:
            target = int(options["target"])
        elif isinstance(self.target_distribution, tuple) and len(self.target_distribution) == 2:
            low, high = self.target_distribution
            target = int(self._rng.integers(low, high + 1))
        elif isinstance(self.target_distribution, (list, np.ndarray)):
            target = int(self._rng.choice(self.target_distribution))

        # Reset simulator
        obs = self.simulator.reset(target=target, initial_wickets=0, random_seed=seed)

        # If shaped reward is active, compute initial win probability
        if self.reward_type == "shaped" and self.win_prob_model is not None:
            self._current_wp = self._estimate_wp()

        info = self._get_info()
        return obs, info

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """ Simulate one delivery given the captain's aggression action """
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}; must be in {self.action_space}")

        # Simulator step
        obs, sim_reward, done, info = self.simulator.step(action)
        terminated = done
        truncated = False

        # Calculate reward
        if self.reward_type == "sparse":
            reward = float(sim_reward)
        else:
            # Potential-based reward shaping (Ng et al. 1999):
            #   F(s, a, s') = gamma * Phi(s') - Phi(s)
            # At a terminal step, Phi(s') is defined as the absorbing-state potential:
            #   1.0 for a win, 0.0 for a loss — no model call at the absorbing state.
            # Combined reward = sparse_terminal + F (sparse_terminal = 0 for non-terminal).
            if terminated:
                # Terminal potential: win → 1.0, loss → 0.0
                next_wp = 1.0 if self.simulator.won else 0.0
            else:
                next_wp = self._estimate_wp()

            # F(s, a, s') = gamma * Phi(s') - Phi(s)
            shaping = self.gamma * next_wp - self._current_wp
            reward = float(sim_reward) + float(shaping)
            self._current_wp = next_wp

        info.update(self._get_info())
        return obs, reward, terminated, truncated, info

    # Helper methods
    def _estimate_wp(self) -> float:
        """Estimate current win probability using the fitted WinProbabilityModel."""
        if self.win_prob_model is None:
            return 0.5

        balls_remaining = MAX_LEGAL_BALLS - self.simulator.legal_balls
        runs_required = max(0, self.simulator.target - self.simulator.score)
        over_number = self.simulator.legal_balls // 6

        if balls_remaining <= 0:
            rrr = 36.0
        else:
            rrr = (runs_required * 6.0) / balls_remaining

        return self.win_prob_model.predict_proba(
            over_number=float(over_number),
            wickets_before=float(self.simulator.wickets),
            runs_required=float(runs_required),
            balls_remaining=float(balls_remaining),
            required_run_rate=float(rrr),
        )

    def _get_info(self) -> dict[str, Any]:
        """Diagnostic metadata."""
        return {
            "score": self.simulator.score,
            "wickets": self.simulator.wickets,
            "legal_balls": self.simulator.legal_balls,
            "balls_remaining": MAX_LEGAL_BALLS - self.simulator.legal_balls,
            "target": self.simulator.target,
            "won": self.simulator.won,
            "is_done": self.simulator.is_done,
            "win_probability": self._current_wp if self.reward_type == "shaped" else None,
        }

    def render(self) -> None:
        """Render current game state to console."""
        print(
            f"Score: {self.simulator.score}/{self.simulator.wickets} | "
            f"Balls: {self.simulator.legal_balls}/120 | "
            f"Target: {self.simulator.target} | "
            f"Need: {max(0, self.simulator.target - self.simulator.score)} off {MAX_LEGAL_BALLS - self.simulator.legal_balls}"
        )
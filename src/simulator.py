

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from outcome_model import DeliveryOutcome, OutcomeModel

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

MAX_LEGAL_BALLS = 120   # 20 overs × 6 balls per over
MAX_WICKETS     = 10    # all-out condition

# Phase boundaries (overs, 0-indexed).
POWERPLAY_END_OVER = 5   # overs 0–5  (legal balls 0–35)
MIDDLE_END_OVER    = 14  # overs 6–14 (legal balls 36–89)
# Death: overs 15–19                   (legal balls 90–119)


# ─────────────────────────────────────────────────────────────────────────────
# SimulatorState (lightweight snapshot for history)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BallRecord:
   
    ball_number: int          # 1-indexed across all deliveries (including extras)
    legal_ball_number: int    # 1-indexed legal balls only
    phase: str
    aggression: int
    outcome: DeliveryOutcome
    score_after: int
    wickets_after: int
    legal_balls_after: int
    bowler_name: Optional[str] = None
    bowling_aggression: int = 1


# ─────────────────────────────────────────────────────────────────────────────
# T20Simulator
# ─────────────────────────────────────────────────────────────────────────────

class T20Simulator:

    def __init__(
        self,
        outcome_model: OutcomeModel,
        random_seed: Optional[int] = 42,
    ):
        if not outcome_model._is_fitted:
            raise ValueError(
                "OutcomeModel must be fitted before passing to T20Simulator. "
                "Call model.fit(train_data) first."
            )
        self._model = outcome_model

        # Internal random generator; reset() will re-seed it.
        self._seed = random_seed
        self._rng: np.random.Generator = np.random.default_rng(seed=random_seed)

        # Match state (initialised by reset()).
        self.target: int = 0
        self.score: int = 0
        self.wickets: int = 0
        self.legal_balls: int = 0
        self.is_done: bool = True     # True until reset() is called.
        self.won: Optional[bool] = None
        self.last_outcome: Optional[DeliveryOutcome] = None

        # Ball-by-ball history (optional; useful for validation).
        self.history: list[BallRecord] = []

        # Running delivery counter (includes extras, for history records).
        self._total_balls: int = 0

    # ─────────────────────────────────────────────────────────────────────────
    # reset()
    # ─────────────────────────────────────────────────────────────────────────

    def reset(
        self,
        target: int,
        initial_wickets: int = 0,
        random_seed: Optional[int] = None,
    ) -> np.ndarray:
       
        if target <= 0:
            raise ValueError(f"target must be positive, got {target}")
        if not (0 <= initial_wickets <= 9):
            raise ValueError(
                f"initial_wickets must be in [0, 9], got {initial_wickets}"
            )

        # Re-seed if requested.
        if random_seed is not None:
            self._rng = np.random.default_rng(seed=random_seed)
        elif self._seed is not None:
            # Re-seed to original seed so each episode is reproducible.
            self._rng = np.random.default_rng(seed=self._seed)
        # else: no seed → continue from previous RNG state (for Monte Carlo).

        self.target       = int(target)
        self.score        = 0
        self.wickets      = int(initial_wickets)
        self.legal_balls  = 0
        self.is_done      = False
        self.won          = None
        self.last_outcome = None
        self.history      = []
        self._total_balls = 0

        return self._observation()

    # ─────────────────────────────────────────────────────────────────────────
    # step()
    # ─────────────────────────────────────────────────────────────────────────

    def step(
        self,
        aggression: int,
        bowling_aggression: int = 1,
        bowler = None,
    ) -> tuple[np.ndarray, float, bool, dict]:
       
        if self.is_done:
            raise RuntimeError(
                "Innings is over. Call reset() to start a new episode."
            )
        if aggression not in (0, 1, 2):
            raise ValueError(
                f"aggression must be 0, 1, or 2; got {aggression}"
            )
        if bowling_aggression not in (0, 1, 2):
            raise ValueError(
                f"bowling_aggression must be 0, 1, or 2; got {bowling_aggression}"
            )

        # ── Determine current match context ──────────────────────────────
        phase = self._current_phase()
        wickets_in_hand = MAX_WICKETS - self.wickets

        # ── Sample one delivery from the outcome model ────────────────────
        try:
            outcome = self._model.sample(
                phase=phase,
                wickets_in_hand=wickets_in_hand,
                aggression=aggression,
                rng=self._rng,
                bowling_aggression=bowling_aggression,
            )
        except TypeError:
            # Fallback for mock test models that only define sample(phase, wih, aggression, rng)
            outcome = self._model.sample(
                phase=phase,
                wickets_in_hand=wickets_in_hand,
                aggression=aggression,
                rng=self._rng,
            )
        self.last_outcome = outcome
        self._total_balls += 1

        # ── Update score ──────────────────────────────────────────────────
        self.score += outcome.total_runs

        # ── Update wickets (only on legal deliveries in our model) ────────
        if outcome.is_legal and outcome.is_wicket:
            self.wickets += 1

        # ── Update legal-ball counter ─────────────────────────────────────
        if outcome.is_legal:
            self.legal_balls += 1

        # ── Update bowler statistics if bowler provided ───────────────────
        bowler_name = None
        if bowler is not None:
            bowler_name = getattr(bowler, "name", str(bowler))
            if hasattr(bowler, "record_delivery"):
                bowler.record_delivery(
                    total_runs=outcome.total_runs,
                    is_wicket=(outcome.is_legal and outcome.is_wicket),
                    is_legal=outcome.is_legal,
                    extra_runs=outcome.extra_runs,
                )

        # ── Append to history ─────────────────────────────────────────────
        self.history.append(
            BallRecord(
                ball_number=self._total_balls,
                legal_ball_number=self.legal_balls,
                phase=phase,
                aggression=aggression,
                outcome=outcome,
                score_after=self.score,
                wickets_after=self.wickets,
                legal_balls_after=self.legal_balls,
                bowler_name=bowler_name,
                bowling_aggression=bowling_aggression,
            )
        )

        # ── Check termination conditions ──────────────────────────────────
        reward = 0.0

        if self.score >= self.target:
            self.is_done = True
            self.won = True
            reward = 1.0

        elif self.wickets >= MAX_WICKETS:
            self.is_done = True
            self.won = False
            reward = -1.0

        elif self.legal_balls >= MAX_LEGAL_BALLS:
            self.is_done = True
            self.won = False
            reward = -1.0

        info = self._info()
        if bowler_name is not None:
            info["bowler_name"] = bowler_name
            info["bowling_aggression"] = bowling_aggression

        return self._observation(), reward, self.is_done, info

    # ─────────────────────────────────────────────────────────────────────────
    # run_episode()  — convenience helper
    # ─────────────────────────────────────────────────────────────────────────

    def run_episode(
        self,
        target: int,
        aggression: int = 1,
        random_seed: Optional[int] = None,
    ) -> dict:
    
        self.reset(target=target, random_seed=random_seed)
        while not self.is_done:
            self.step(aggression=aggression)
        return {
            "score":             self.score,
            "wickets":           self.wickets,
            "legal_balls":       self.legal_balls,
            "won":               self.won,
            "total_deliveries":  self._total_balls,
            "history":           self.history,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _current_phase(self) -> str:
        over = self.legal_balls // 6
        if over <= POWERPLAY_END_OVER:
            return "powerplay"
        elif over <= MIDDLE_END_OVER:
            return "middle"
        else:
            return "death"

    def _observation(self) -> np.ndarray:
        """Build and return the current state observation vector."""
        # Import here to avoid circular import at module load.
        from match_state import MatchState, state_to_observation

        balls_remaining = MAX_LEGAL_BALLS - self.legal_balls
        runs_required   = self.target - self.score

        if balls_remaining == 0:
            rrr = MatchState._RRR_SENTINEL
        elif runs_required <= 0:
            rrr = 0.0
        else:
            rrr = (runs_required * 6.0) / balls_remaining

        state = MatchState(
            current_score=self.score,
            wickets_lost=self.wickets,
            balls_remaining=balls_remaining,
            target=float(self.target),
            runs_required=float(runs_required),
            required_run_rate=rrr,
        )
        return state_to_observation(state)

    def _info(self) -> dict:
        """Return diagnostic info dict."""
        return {
            "score":           self.score,
            "wickets":         self.wickets,
            "legal_balls":     self.legal_balls,
            "balls_remaining": MAX_LEGAL_BALLS - self.legal_balls,
            "won":             self.won,
            "last_outcome":    self.last_outcome,
        }

    @property
    def balls_remaining(self) -> int:
        """Legal deliveries remaining in the 120-ball innings."""
        return max(0, MAX_LEGAL_BALLS - self.legal_balls)

    @property
    def runs_required(self) -> int:
        """Runs required to reach or exceed target."""
        return max(0, self.target - self.score)

    @property
    def required_run_rate(self) -> float:
        """Required runs per 6 legal balls."""
        br = self.balls_remaining
        rr = self.runs_required
        if br == 0:
            return 36.0 if rr > 0 else 0.0
        return (rr * 6.0) / br

    @property
    def current_run_rate(self) -> float:
        """Current runs per 6 legal balls bowled."""
        if self.legal_balls == 0:
            return 0.0
        return (self.score * 6.0) / self.legal_balls

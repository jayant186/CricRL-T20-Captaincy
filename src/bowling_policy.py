
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"

# ─────────────────────────────────────────────────────────────────────────────
# Bowling Actions & Modifiers
# ─────────────────────────────────────────────────────────────────────────────

BOWLING_ACTIONS = {
    0: "Defensive",
    1: "Balanced",
    2: "Attacking",
}

BOWLING_ACTION_COLORS = {
    0: "#2ca02c",  # Green / Containment
    1: "#1f77b4",  # Blue / Balanced
    2: "#d62728",  # Red / Attacking
}

# Empirical tactical multipliers applied jointly with batting aggression
# Defensive bowling: suppresses boundary rate (0.80x), slightly lower wicket rate (0.88x)
# Balanced bowling: baseline 1.0x
# Attacking bowling: hunts wickets (1.25x), higher boundary risk (1.15x)
BOWLING_BOUNDARY_MULT = {0: 0.80, 1: 1.00, 2: 1.15}
BOWLING_WICKET_MULT   = {0: 0.88, 1: 1.00, 2: 1.25}


# ─────────────────────────────────────────────────────────────────────────────
# Bowler Dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Bowler:
    """
    Representation of an individual bowler in a match innings.
    """
    name: str
    team: str
    role: str               # "Opening Pacer", "Middle Spinner", "Death Specialist", "All-Rounder"
    bowling_type: str        # e.g., "Right-arm Fast", "Right-arm Leg-break"
    career_econ: float       # Career IPL economy rate (runs per over)
    career_sr: float         # Career balls per wicket
    powerplay_econ: float    # Historical economy in overs 0-5
    middle_econ: float       # Historical economy in overs 6-14
    death_econ: float        # Historical economy in overs 15-19
    max_overs: int = 4

    # Match innings state
    legal_balls: int = 0
    runs_conceded: int = 0
    wickets_taken: int = 0
    extras_conceded: int = 0
    maidens: int = 0
    current_over_runs: int = 0

    @property
    def completed_overs(self) -> int:
        return self.legal_balls // 6

    @property
    def balls_in_over(self) -> int:
        return self.legal_balls % 6

    @property
    def overs_bowled_str(self) -> str:
        return f"{self.completed_overs}.{self.balls_in_over}"

    @property
    def overs_remaining(self) -> int:
        return max(0, self.max_overs - self.completed_overs)

    @property
    def legal_balls_remaining(self) -> int:
        return max(0, (self.max_overs * 6) - self.legal_balls)

    @property
    def is_maxed_out(self) -> bool:
        return self.legal_balls >= (self.max_overs * 6)

    @property
    def economy_rate(self) -> float:
        if self.legal_balls == 0:
            return 0.0
        return (self.runs_conceded * 6.0) / self.legal_balls

    @property
    def figures_str(self) -> str:
        return f"{self.overs_bowled_str}-{self.maidens}-{self.runs_conceded}-{self.wickets_taken} (Econ: {self.economy_rate:.2f})"

    def record_delivery(self, total_runs: int, is_wicket: bool, is_legal: bool, extra_runs: int = 0) -> None:
        """Update bowler statistics following a delivery."""
        self.runs_conceded += total_runs
        self.current_over_runs += total_runs
        if extra_runs > 0:
            self.extras_conceded += extra_runs
        if is_wicket:
            self.wickets_taken += 1
        if is_legal:
            self.legal_balls += 1

    def finish_over(self) -> None:
        """Call at completion of an over to update maiden status."""
        if self.current_over_runs == 0 and self.balls_in_over == 0 and self.legal_balls > 0:
            self.maidens += 1
        self.current_over_runs = 0

    def reset(self) -> None:
        """Reset bowler match figures."""
        self.legal_balls = 0
        self.runs_conceded = 0
        self.wickets_taken = 0
        self.extras_conceded = 0
        self.maidens = 0
        self.current_over_runs = 0


# ─────────────────────────────────────────────────────────────────────────────
# Bowling Squad
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BowlingSquad:
    """
    Manages the bowling team's roster and enforces over limit constraints.
    """
    team_name: str
    bowlers: List[Bowler]
    last_bowler_name: Optional[str] = None
    over_bowler_history: List[str] = field(default_factory=list)

    def get_bowler(self, name: str) -> Optional[Bowler]:
        for b in self.bowlers:
            if b.name.strip().lower() == name.strip().lower():
                return b
        return None

    def get_eligible_bowlers(self, over_number: int) -> List[Bowler]:
        
        eligible = []
        for b in self.bowlers:
            if b.is_maxed_out:
                continue
            if self.last_bowler_name and b.name == self.last_bowler_name:
                continue
            eligible.append(b)

        # Extreme fallback: if due to small squad or irregular inputs no bowler meets
        # both criteria, allow any bowler who hasn't reached 24 legal balls.
        if not eligible:
            eligible = [b for b in self.bowlers if not b.is_maxed_out]

        return eligible

    def start_over(self, bowler_name: str) -> Bowler:
        """Start a new over with specified bowler."""
        bowler = self.get_bowler(bowler_name)
        if bowler is None:
            raise ValueError(f"Bowler {bowler_name} not found in squad {self.team_name}")
        if not self.over_bowler_history or self.over_bowler_history[-1] != bowler.name:
            self.over_bowler_history.append(bowler.name)
        return bowler

    def complete_over(self, bowler_name: str) -> None:
        """Mark completion of an over and enforce consecutive-over restriction."""
        bowler = self.get_bowler(bowler_name)
        if bowler:
            bowler.finish_over()
        self.last_bowler_name = bowler_name
        if bowler and (not self.over_bowler_history or self.over_bowler_history[-1] != bowler.name):
            self.over_bowler_history.append(bowler.name)

    def reset(self) -> None:
        """Reset all bowlers and match history in the squad."""
        self.last_bowler_name = None
        self.over_bowler_history = []
        for b in self.bowlers:
            b.reset()

    def get_figures_summary(self) -> pd.DataFrame:
        """Return formatted bowling scorecard DataFrame."""
        records = []
        for b in self.bowlers:
            if b.legal_balls > 0 or b.runs_conceded > 0:
                records.append({
                    "Bowler": b.name,
                    "Role": b.role,
                    "Overs": b.overs_bowled_str,
                    "Maidens": b.maidens,
                    "Runs": b.runs_conceded,
                    "Wickets": b.wickets_taken,
                    "Economy": round(b.economy_rate, 2),
                    "Overs Left": b.overs_remaining,
                })
        return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# Pre-Configured Authentic IPL Squads
# ─────────────────────────────────────────────────────────────────────────────

def get_csk_squad() -> BowlingSquad:
    """Chennai Super Kings canonical bowling squad."""
    bowlers = [
        Bowler("DL Chahar", "Chennai Super Kings", "Opening Pacer", "Right-arm Medium-Fast", 7.99, 21.5, 8.15, 8.44, 10.19),
        Bowler("RA Jadeja", "Chennai Super Kings", "Middle Spinner", "Slow Left-arm Orthodox", 7.70, 24.2, 7.80, 7.42, 8.85),
        Bowler("R Ashwin", "Chennai Super Kings", "Middle Spinner", "Right-arm Off-break", 6.83, 23.4, 6.95, 6.70, 7.90),
        Bowler("SN Thakur", "Chennai Super Kings", "Strike Pacer", "Right-arm Medium-Fast", 9.19, 19.1, 8.80, 8.95, 10.25),
        Bowler("DJ Bravo", "Chennai Super Kings", "Death Specialist", "Right-arm Medium-Fast", 8.50, 16.8, 8.00, 7.70, 9.52),
    ]
    return BowlingSquad("Chennai Super Kings", bowlers)


def get_mi_squad() -> BowlingSquad:
    """Mumbai Indians canonical bowling squad."""
    bowlers = [
        Bowler("JJ Bumrah", "Mumbai Indians", "Death Specialist", "Right-arm Fast", 7.53, 17.5, 7.14, 6.75, 8.39),
        Bowler("SL Malinga", "Mumbai Indians", "Death Specialist", "Right-arm Fast (Sling)", 7.40, 15.0, 6.54, 7.50, 8.20),
        Bowler("Harbhajan Singh", "Mumbai Indians", "Middle Spinner", "Right-arm Off-break", 7.10, 21.4, 6.80, 7.05, 8.10),
        Bowler("MJ McClenaghan", "Mumbai Indians", "Opening Pacer", "Left-arm Fast-Medium", 8.66, 17.0, 8.25, 8.70, 9.40),
        Bowler("HH Pandya", "Mumbai Indians", "All-Rounder", "Right-arm Medium-Fast", 9.79, 18.8, 9.10, 9.20, 10.50),
    ]
    return BowlingSquad("Mumbai Indians", bowlers)


def get_rcb_squad() -> BowlingSquad:
    """Royal Challengers Bengaluru canonical bowling squad."""
    bowlers = [
        Bowler("Mohammed Siraj", "Royal Challengers Bengaluru", "Opening Pacer", "Right-arm Fast", 8.84, 21.4, 8.20, 8.80, 10.10),
        Bowler("YS Chahal", "Royal Challengers Bengaluru", "Middle Spinner", "Right-arm Leg-break", 7.74, 16.9, 7.99, 7.88, 9.61),
        Bowler("HV Patel", "Royal Challengers Bengaluru", "Death Specialist", "Right-arm Medium-Fast", 8.68, 15.0, 8.40, 8.30, 9.35),
        Bowler("Z Khan", "Royal Challengers Bengaluru", "Opening Pacer", "Left-arm Fast", 8.08, 18.2, 7.70, 7.95, 8.90),
        Bowler("JR Hazlewood", "Royal Challengers Bengaluru", "Strike Pacer", "Right-arm Fast-Medium", 8.95, 14.8, 7.85, 8.50, 9.80),
    ]
    return BowlingSquad("Royal Challengers Bengaluru", bowlers)


def get_kkr_squad() -> BowlingSquad:
    """Kolkata Knight Riders canonical bowling squad."""
    bowlers = [
        Bowler("UT Yadav", "Kolkata Knight Riders", "Opening Pacer", "Right-arm Fast", 8.71, 19.3, 8.10, 8.80, 10.05),
        Bowler("SP Narine", "Kolkata Knight Riders", "Middle Spinner", "Right-arm Off-break", 6.93, 20.3, 7.03, 6.68, 7.41),
        Bowler("CV Varun", "Kolkata Knight Riders", "Middle Spinner", "Right-arm Leg-break / Mystery", 7.83, 18.7, 7.60, 7.55, 8.65),
        Bowler("AD Russell", "Kolkata Knight Riders", "Death Specialist", "Right-arm Fast", 9.64, 15.2, 8.90, 9.10, 10.40),
        Bowler("PP Chawla", "Kolkata Knight Riders", "Middle Spinner", "Right-arm Leg-break", 8.31, 19.7, 8.10, 8.20, 9.20),
    ]
    return BowlingSquad("Kolkata Knight Riders", bowlers)


def get_custom_squad(team_name: str = "Custom Team") -> BowlingSquad:
    """Generic 5-bowler squad for custom teams."""
    bowlers = [
        Bowler("Pacer 1 (Lead Opener)", team_name, "Opening Pacer", "Right-arm Fast", 7.80, 20.0, 7.50, 8.00, 9.20),
        Bowler("Pacer 2 (Strike Pacer)", team_name, "Strike Pacer", "Left-arm Fast-Medium", 8.50, 18.0, 8.20, 8.40, 9.60),
        Bowler("Spinner 1 (Off-Break)", team_name, "Middle Spinner", "Right-arm Off-break", 7.10, 22.0, 7.20, 6.90, 8.10),
        Bowler("Spinner 2 (Leg-Break)", team_name, "Middle Spinner", "Right-arm Leg-break", 7.90, 17.5, 8.00, 7.60, 9.30),
        Bowler("Death Pacer (Yorker)", team_name, "Death Specialist", "Right-arm Fast", 8.40, 16.0, 8.10, 8.00, 8.80),
    ]
    return BowlingSquad(team_name, bowlers)


AVAILABLE_SQUADS = {
    "Chennai Super Kings": get_csk_squad,
    "Mumbai Indians": get_mi_squad,
    "Royal Challengers Bengaluru": get_rcb_squad,
    "Kolkata Knight Riders": get_kkr_squad,
}


# ─────────────────────────────────────────────────────────────────────────────
# Bowling Captaincy Policy Interface & Implementations
# ─────────────────────────────────────────────────────────────────────────────

class BaseBowlingCaptaincy(ABC):
    """Abstract base class for bowling captaincy decisions."""

    @abstractmethod
    def select_bowler(
        self,
        squad: BowlingSquad,
        legal_balls_bowled: int,
        score: int,
        wickets: int,
        target: int,
    ) -> Tuple[Bowler, str]:
        """
        Choose the bowler for the upcoming over.
        Returns (Bowler, justification_reason).
        """
        pass

    @abstractmethod
    def select_tactic(
        self,
        bowler: Bowler,
        legal_balls_bowled: int,
        score: int,
        wickets: int,
        target: int,
    ) -> Tuple[int, str]:
        """
        Choose bowling tactic (0=Defensive, 1=Balanced, 2=Attacking).
        Returns (action_int, justification_reason).
        """
        pass


class PhaseSpecialistCaptaincy(BaseBowlingCaptaincy):
    """
    Standard IPL phase-specialized captaincy policy:
    - Overs 0–5 (Powerplay): Opening pacers with low powerplay economy.
    - Overs 6–14 (Middle): Middle spinners and containment options.
    - Overs 15–19 (Death): Primary death specialists and yorker pacers.
    - Preserves death overs for specialists so death overs aren't bowled by part-timers.
    """

    def select_bowler(
        self,
        squad: BowlingSquad,
        legal_balls_bowled: int,
        score: int,
        wickets: int,
        target: int,
    ) -> Tuple[Bowler, str]:
        current_over = legal_balls_bowled // 6
        eligible = squad.get_eligible_bowlers(current_over)
        if not eligible:
            raise RuntimeError(f"No eligible bowler available in squad {squad.team_name} for over {current_over}")

        # Phase determination
        if current_over < 6:
            phase = "powerplay"
            preferred_roles = ["Opening Pacer", "Strike Pacer"]
        elif current_over < 15:
            phase = "middle"
            preferred_roles = ["Middle Spinner", "All-Rounder", "Opening Pacer"]
        else:
            phase = "death"
            preferred_roles = ["Death Specialist", "Strike Pacer"]

        # In powerplay and middle phases, protect death specialists from bowling out completely
        overs_to_death = max(0, 15 - current_over)
        death_specialists = [b for b in squad.bowlers if "Death" in b.role]

        candidates = []
        for b in eligible:
            # If bowler is a death specialist and we have death overs coming up, ensure they keep >= 1-2 overs
            if phase != "death" and b in death_specialists:
                if b.completed_overs >= 2 and overs_to_death > 0:
                    continue  # preserve for death
            candidates.append(b)

        if not candidates:
            candidates = eligible

        # Score candidates based on phase-specific economy and role match
        def rank_score(b: Bowler) -> float:
            score_val = 0.0
            if b.role in preferred_roles:
                score_val += 10.0
            if phase == "powerplay":
                score_val -= b.powerplay_econ
            elif phase == "middle":
                score_val -= b.middle_econ
            else:
                score_val -= b.death_econ
            return score_val

        best_bowler = max(candidates, key=rank_score)
        reason = (
            f"Phase specialist ({phase.title()} Over {current_over + 1}): Selected {best_bowler.name} "
            f"({best_bowler.role}, {phase} econ: {getattr(best_bowler, f'{phase}_econ'):.2f}, "
            f"{best_bowler.overs_remaining} overs left)."
        )
        return best_bowler, reason

    def select_tactic(
        self,
        bowler: Bowler,
        legal_balls_bowled: int,
        score: int,
        wickets: int,
        target: int,
    ) -> Tuple[int, str]:
        current_over = legal_balls_bowled // 6
        balls_left = max(0, 120 - legal_balls_bowled)
        runs_needed = max(0, target - score)
        rrr = (runs_needed * 6.0) / balls_left if balls_left > 0 else 0.0

        if current_over < 6:
            # Powerplay: Attack to get early wickets
            return 2, "Powerplay tactical aggression: Attacking fields and searching for early wickets."
        elif current_over >= 15:
            # Death overs: If RRR is very high, play defensive containment; otherwise balanced/yorkers
            if rrr > 11.0:
                return 0, f"Death overs containment: Chasing team needs RRR {rrr:.2f}; bowling defensive yorkers."
            return 1, "Death overs execution: Balanced yorker-length attack."
        else:
            # Middle overs
            if wickets >= 6:
                return 2, f"Tail-ender attack: Batting team {wickets} wickets down; attacking to wrap up innings."
            elif rrr < 7.0:
                return 2, f"Partnership break: Chasing team cruising at RRR {rrr:.2f}; attacking to induce mistimed shot."
            elif rrr > 10.0:
                return 0, f"Run-rate choke: RRR {rrr:.2f} is high; bowling defensive containment."
            return 1, "Middle overs control: Standard balanced tactical bowling."


class RiskAdaptiveCaptaincy(BaseBowlingCaptaincy):
    """
    Dynamic reactive bowling captaincy:
    - Reads real-time match pressure (RRR, balls left, wickets fallen).
    - If RRR < 7.0 or batting team has 7+ wickets in hand: summons strike bowlers on Attacking mode.
    - If RRR > 10.5: brings on lowest-economy bowlers on Defensive mode to choke the chase.
    - In death overs: deploys death specialists.
    """

    def select_bowler(
        self,
        squad: BowlingSquad,
        legal_balls_bowled: int,
        score: int,
        wickets: int,
        target: int,
    ) -> Tuple[Bowler, str]:
        current_over = legal_balls_bowled // 6
        eligible = squad.get_eligible_bowlers(current_over)
        balls_left = max(0, 120 - legal_balls_bowled)
        runs_needed = max(0, target - score)
        rrr = (runs_needed * 6.0) / balls_left if balls_left > 0 else 0.0

        # Preserve death specialist quota for overs 15-19
        overs_to_death = max(0, 15 - current_over)
        death_specialists = [b for b in squad.bowlers if "Death" in b.role]

        candidates = []
        for b in eligible:
            if current_over < 15 and b in death_specialists:
                if b.completed_overs >= 2 and overs_to_death > 0:
                    continue  # preserve for death
            candidates.append(b)
        if not candidates:
            candidates = eligible

        if current_over >= 15:
            # Death phase: Pick best death economy
            best = min(candidates, key=lambda b: b.death_econ)
            reason = f"Risk-Adaptive Death: Selected {best.name} (best death economy: {best.death_econ:.2f})."
            return best, reason

        if rrr < 7.5 or wickets <= 2:
            # Need wickets urgently: pick bowler with best strike rate
            best = min(candidates, key=lambda b: b.career_sr)
            reason = f"Risk-Adaptive Wicket Hunting: RRR {rrr:.2f} is low / only {wickets} down; selected strike bowler {best.name} (SR: {best.career_sr:.1f})."
            return best, reason
        else:
            # Containment: pick bowler with lowest phase economy
            phase_attr = "powerplay_econ" if current_over < 6 else "middle_econ"
            best = min(candidates, key=lambda b: getattr(b, phase_attr))
            reason = f"Risk-Adaptive Containment: RRR {rrr:.2f} is high; selected containment bowler {best.name} (econ: {getattr(best, phase_attr):.2f})."
            return best, reason

    def select_tactic(
        self,
        bowler: Bowler,
        legal_balls_bowled: int,
        score: int,
        wickets: int,
        target: int,
    ) -> Tuple[int, str]:
        balls_left = max(0, 120 - legal_balls_bowled)
        runs_needed = max(0, target - score)
        rrr = (runs_needed * 6.0) / balls_left if balls_left > 0 else 0.0

        if wickets >= 7 or rrr < 7.0:
            return 2, f"Attacking approach: Prioritizing dismissals (RRR: {rrr:.2f}, Wickets lost: {wickets})."
        elif rrr > 10.0:
            return 0, f"Defensive approach: Prioritizing dot balls and boundary denial (RRR: {rrr:.2f})."
        return 1, "Balanced approach: Standard tactical execution."


class RoundRobinCaptaincy(BaseBowlingCaptaincy):
    """
    Baseline policy: Simple rotating cycle through eligible bowlers with Balanced tactics.
    """

    def select_bowler(
        self,
        squad: BowlingSquad,
        legal_balls_bowled: int,
        score: int,
        wickets: int,
        target: int,
    ) -> Tuple[Bowler, str]:
        current_over = legal_balls_bowled // 6
        eligible = squad.get_eligible_bowlers(current_over)
        # Select bowler with least balls bowled
        best = min(eligible, key=lambda b: (b.legal_balls, b.runs_conceded))
        reason = f"Round-Robin Cycle: Selected {best.name} ({best.legal_balls} balls bowled so far)."
        return best, reason

    def select_tactic(
        self,
        bowler: Bowler,
        legal_balls_bowled: int,
        score: int,
        wickets: int,
        target: int,
    ) -> Tuple[int, str]:
        return 1, "Round-robin baseline: Balanced bowling."


class AlwaysAttackingCaptaincy(BaseBowlingCaptaincy):
    """Heuristic baseline: Always bowls Attacking."""

    def select_bowler(self, squad: BowlingSquad, legal_balls_bowled: int, score: int, wickets: int, target: int) -> Tuple[Bowler, str]:
        current_over = legal_balls_bowled // 6
        eligible = squad.get_eligible_bowlers(current_over)
        best = min(eligible, key=lambda b: b.career_sr)
        return best, f"Always-Attacking: Selected strike bowler {best.name} (SR: {best.career_sr:.1f})."

    def select_tactic(self, bowler: Bowler, legal_balls_bowled: int, score: int, wickets: int, target: int) -> Tuple[int, str]:
        return 2, "Always-Attacking baseline: Maximum wicket aggression."


class AlwaysDefensiveCaptaincy(BaseBowlingCaptaincy):
    """Heuristic baseline: Always bowls Defensive."""

    def select_bowler(self, squad: BowlingSquad, legal_balls_bowled: int, score: int, wickets: int, target: int) -> Tuple[Bowler, str]:
        current_over = legal_balls_bowled // 6
        eligible = squad.get_eligible_bowlers(current_over)
        best = min(eligible, key=lambda b: b.career_econ)
        return best, f"Always-Defensive: Selected containment bowler {best.name} (Career Econ: {best.career_econ:.2f})."

    def select_tactic(self, bowler: Bowler, legal_balls_bowled: int, score: int, wickets: int, target: int) -> Tuple[int, str]:
        return 0, "Always-Defensive baseline: Maximum run containment."

import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bowling_policy import (
    Bowler,
    BowlingSquad,
    PhaseSpecialistCaptaincy,
    RiskAdaptiveCaptaincy,
    RoundRobinCaptaincy,
    AlwaysAttackingCaptaincy,
    AlwaysDefensiveCaptaincy,
    get_csk_squad,
    get_mi_squad,
    get_rcb_squad,
    get_kkr_squad,
    get_custom_squad,
    AVAILABLE_SQUADS,
    BOWLING_ACTIONS,
)
from outcome_model import DeliveryOutcome, OutcomeModel
from simulator import T20Simulator


@pytest.fixture
def sample_bowler() -> Bowler:
    return Bowler(
        name="JJ Bumrah",
        team="Mumbai Indians",
        role="Death Specialist",
        bowling_type="Right-arm Fast",
        career_econ=7.53,
        career_sr=17.5,
        powerplay_econ=7.14,
        middle_econ=6.75,
        death_econ=8.39,
    )


@pytest.fixture
def csk_squad() -> BowlingSquad:
    return get_csk_squad()


@pytest.fixture
def fitted_outcome_model() -> OutcomeModel:
    inn2_path = ROOT / "data" / "processed" / "innings2_chase.csv"
    df = pd.read_csv(inn2_path, low_memory=False)
    return OutcomeModel().fit(df)



# 1. Bowler Unit Tests

class TestBowler:
    def test_initial_state(self, sample_bowler: Bowler):
        assert sample_bowler.legal_balls == 0
        assert sample_bowler.runs_conceded == 0
        assert sample_bowler.wickets_taken == 0
        assert sample_bowler.extras_conceded == 0
        assert sample_bowler.completed_overs == 0
        assert sample_bowler.balls_in_over == 0
        assert sample_bowler.overs_remaining == 4
        assert not sample_bowler.is_maxed_out
        assert sample_bowler.economy_rate == 0.0

    def test_record_legal_ball(self, sample_bowler: Bowler):
        sample_bowler.record_delivery(total_runs=4, is_wicket=False, is_legal=True)
        assert sample_bowler.legal_balls == 1
        assert sample_bowler.runs_conceded == 4
        assert sample_bowler.wickets_taken == 0
        assert sample_bowler.economy_rate == 24.0

    def test_record_wicket(self, sample_bowler: Bowler):
        sample_bowler.record_delivery(total_runs=0, is_wicket=True, is_legal=True)
        assert sample_bowler.legal_balls == 1
        assert sample_bowler.runs_conceded == 0
        assert sample_bowler.wickets_taken == 1
        assert sample_bowler.economy_rate == 0.0

    def test_record_wide_extra(self, sample_bowler: Bowler):
        # Wide does not increment legal_balls but adds 1 run and 1 extra
        sample_bowler.record_delivery(total_runs=1, is_wicket=False, is_legal=False, extra_runs=1)
        assert sample_bowler.legal_balls == 0
        assert sample_bowler.runs_conceded == 1
        assert sample_bowler.extras_conceded == 1

    def test_max_overs_flag(self, sample_bowler: Bowler):
        for _ in range(24):
            sample_bowler.record_delivery(total_runs=1, is_wicket=False, is_legal=True)
        assert sample_bowler.completed_overs == 4
        assert sample_bowler.overs_remaining == 0
        assert sample_bowler.is_maxed_out

    def test_maiden_over(self, sample_bowler: Bowler):
        for _ in range(6):
            sample_bowler.record_delivery(total_runs=0, is_wicket=False, is_legal=True)
        sample_bowler.finish_over()
        assert sample_bowler.maidens == 1
        assert sample_bowler.completed_overs == 1


# 2. Bowling Squad Unit Tests

class TestBowlingSquad:
    def test_squad_creation(self, csk_squad: BowlingSquad):
        assert csk_squad.team_name == "Chennai Super Kings"
        assert len(csk_squad.bowlers) == 5
        assert csk_squad.last_bowler_name is None

    def test_initial_eligibility(self, csk_squad: BowlingSquad):
        eligible = csk_squad.get_eligible_bowlers(over_number=0)
        assert len(eligible) == 5

    def test_consecutive_over_rule(self, csk_squad: BowlingSquad):
        # Bowler DL Chahar bowls over 0
        csk_squad.start_over("DL Chahar")
        csk_squad.complete_over("DL Chahar")
        assert csk_squad.last_bowler_name == "DL Chahar"

        # DL Chahar cannot bowl over 1
        eligible_over_1 = csk_squad.get_eligible_bowlers(over_number=1)
        names = [b.name for b in eligible_over_1]
        assert "DL Chahar" not in names
        assert len(eligible_over_1) == 4

        # After another bowler bowls over 1, DL Chahar is eligible for over 2
        csk_squad.start_over("RA Jadeja")
        csk_squad.complete_over("RA Jadeja")
        eligible_over_2 = csk_squad.get_eligible_bowlers(over_number=2)
        names_2 = [b.name for b in eligible_over_2]
        assert "DL Chahar" in names_2
        assert "RA Jadeja" not in names_2

    def test_max_overs_exclusion(self, csk_squad: BowlingSquad):
        bowler = csk_squad.get_bowler("DL Chahar")
        assert bowler is not None
        # Simulate 24 legal balls (4 overs)
        for _ in range(24):
            bowler.record_delivery(total_runs=1, is_wicket=False, is_legal=True)

        assert bowler.is_maxed_out
        eligible = csk_squad.get_eligible_bowlers(over_number=10)
        names = [b.name for b in eligible]
        assert "DL Chahar" not in names

    def test_available_squad_presets(self):
        for name, constructor in AVAILABLE_SQUADS.items():
            squad = constructor()
            assert len(squad.bowlers) == 5
            assert squad.team_name == name


# 3. Bowling Captaincy Policy Tests

class TestBowlingCaptaincyPolicies:
    def test_phase_specialist_powerplay(self, csk_squad: BowlingSquad):
        policy = PhaseSpecialistCaptaincy()
        # Over 0 (legal_balls=0) -> Powerplay
        bowler, reason = policy.select_bowler(csk_squad, legal_balls_bowled=0, score=0, wickets=0, target=165)
        # Should pick Opening Pacer (DL Chahar)
        assert bowler.role in ["Opening Pacer", "Strike Pacer"]
        assert "Powerplay" in reason

    def test_phase_specialist_middle_spinner(self, csk_squad: BowlingSquad):
        policy = PhaseSpecialistCaptaincy()
        # Over 7 (legal_balls=42) -> Middle phase
        # Ensure fast bowlers bowled powerplay
        csk_squad.complete_over("DL Chahar")
        bowler, reason = policy.select_bowler(csk_squad, legal_balls_bowled=42, score=50, wickets=2, target=165)
        assert bowler.role in ["Middle Spinner", "All-Rounder", "Opening Pacer"]
        assert "Middle" in reason

    def test_phase_specialist_death(self, csk_squad: BowlingSquad):
        policy = PhaseSpecialistCaptaincy()
        # Over 18 (legal_balls=108) -> Death phase
        csk_squad.complete_over("RA Jadeja")
        bowler, reason = policy.select_bowler(csk_squad, legal_balls_bowled=108, score=140, wickets=5, target=165)
        # Death specialist (DJ Bravo)
        assert bowler.name == "DJ Bravo" or "Death" in bowler.role
        assert "Death" in reason

    def test_risk_adaptive_attacking_on_low_rrr(self, csk_squad: BowlingSquad):
        policy = RiskAdaptiveCaptaincy()
        # Low RRR: 20 runs needed from 30 balls (RRR = 4.0), 1 wicket down
        tactic, reason = policy.select_tactic(
            csk_squad.bowlers[0],
            legal_balls_bowled=90,
            score=130,
            wickets=1,
            target=150,
        )
        assert tactic == 2  # Attacking
        assert "Attacking" in reason

    def test_risk_adaptive_defensive_on_high_rrr(self, csk_squad: BowlingSquad):
        policy = RiskAdaptiveCaptaincy()
        # High RRR: 60 runs needed from 24 balls (RRR = 15.0)
        tactic, reason = policy.select_tactic(
            csk_squad.bowlers[0],
            legal_balls_bowled=96,
            score=105,
            wickets=4,
            target=165,
        )
        assert tactic == 0  # Defensive
        assert "Defensive" in reason

    def test_round_robin_rotation(self, csk_squad: BowlingSquad):
        policy = RoundRobinCaptaincy()
        bowlers_picked = []
        for over in range(5):
            bowler, _ = policy.select_bowler(csk_squad, over * 6, score=over * 8, wickets=0, target=165)
            bowlers_picked.append(bowler.name)
            # simulate over
            for _ in range(6):
                bowler.record_delivery(1, False, True)
            csk_squad.complete_over(bowler.name)

        # All 5 unique bowlers should have bowled in first 5 overs
        assert len(set(bowlers_picked)) == 5


# 4. End-to-End Match Simulation Integration Tests

class TestBowlingMatchSimulation:
    def test_full_innings_with_bowling_squad(self, fitted_outcome_model: OutcomeModel):
        squad = get_mi_squad()
        captaincy = PhaseSpecialistCaptaincy()
        sim = T20Simulator(outcome_model=fitted_outcome_model, random_seed=42)
        sim.reset(target=170)

        current_bowler = None
        current_tactic = 1

        while not sim.is_done:
            # Over start check
            if sim.legal_balls % 6 == 0 and sim.history and sim.history[-1].outcome.is_legal:
                if current_bowler:
                    squad.complete_over(current_bowler.name)
                current_bowler, _ = captaincy.select_bowler(
                    squad, sim.legal_balls, sim.score, sim.wickets, sim.target
                )
                current_tactic, _ = captaincy.select_tactic(
                    current_bowler, sim.legal_balls, sim.score, sim.wickets, sim.target
                )
            elif current_bowler is None:
                current_bowler, _ = captaincy.select_bowler(
                    squad, sim.legal_balls, sim.score, sim.wickets, sim.target
                )
                current_tactic, _ = captaincy.select_tactic(
                    current_bowler, sim.legal_balls, sim.score, sim.wickets, sim.target
                )

            # Step simulator with batting aggression=1 (Balanced) and bowling tactic
            obs, rew, done, info = sim.step(
                aggression=1,
                bowling_aggression=current_tactic,
                bowler=current_bowler,
            )

        # Post-match verification
        # 1. No bowler bowled > 24 legal balls
        for b in squad.bowlers:
            assert b.legal_balls <= 24, f"Bowler {b.name} exceeded 24 legal balls ({b.legal_balls})"

        # 2. Total legal balls bowled by squad matches simulator legal_balls
        squad_legal = sum(b.legal_balls for b in squad.bowlers)
        assert squad_legal == sim.legal_balls

        # 3. Total wickets taken by squad matches simulator wickets lost
        squad_wickets = sum(b.wickets_taken for b in squad.bowlers)
        assert squad_wickets == sim.wickets

        # 4. Consecutive over rule was strictly respected
        history = squad.over_bowler_history
        for i in range(len(history) - 1):
            assert history[i] != history[i + 1], f"Consecutive overs bowled by {history[i]}!"

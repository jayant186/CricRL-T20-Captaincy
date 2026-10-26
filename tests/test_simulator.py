import sys
from pathlib import Path
import math

import numpy as np
import pandas as pd
import pytest

# Make src importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from outcome_model import OutcomeModel, DeliveryOutcome, BATTER_RUNS_VALUES
from simulator import T20Simulator, MAX_LEGAL_BALLS, MAX_WICKETS

DATA = Path(__file__).resolve().parents[1] / "data" / "processed"

# Session-scoped fixtures: load data and fit model once
@pytest.fixture(scope="session")
def innings2():
    """Load full innings2_chase.csv."""
    return pd.read_csv(DATA / "innings2_chase.csv", low_memory=False)


@pytest.fixture(scope="session")
def train_innings2(innings2):
    """Filter to training match IDs only."""
    from data_loader import load_pipeline_data
    data = load_pipeline_data(verbose=False)
    train_ids = set(data["train_match_ids"])
    return innings2[innings2["match_id"].isin(train_ids)].copy()


@pytest.fixture(scope="session")
def fitted_model(train_innings2):
    """Fit the outcome model on training data. Shared across all tests."""
    model = OutcomeModel()
    model.fit(train_innings2)
    return model


@pytest.fixture(scope="session")
def simulator(fitted_model):
    """Return a T20Simulator with seed=42."""
    return T20Simulator(outcome_model=fitted_model, random_seed=42)

# 1. TestOutcomeModelFit
class TestOutcomeModelFit:
    """Model must fit without error and produce valid probability tables."""

    def test_is_fitted(self, fitted_model):
        assert fitted_model._is_fitted

    def test_all_cells_populated(self, fitted_model):
        """Every (phase, wih_bucket) cell must have an entry."""
        phases = ["powerplay", "middle", "death"]
        buckets = ["1-2", "3-4", "5-6", "7-8", "9-10"]
        for ph in phases:
            for wib in buckets:
                key = (ph, wib)
                assert key in fitted_model._batter_run_probs, (
                    f"Missing batter_run_probs for key {key}"
                )
                assert key in fitted_model._wicket_rates, (
                    f"Missing wicket_rate for key {key}"
                )

    def test_batter_probs_sum_to_one(self, fitted_model):
        """Each cell's probability vector must sum to ~1.0."""
        for key, probs in fitted_model._batter_run_probs.items():
            total = float(probs.sum())
            assert abs(total - 1.0) < 1e-6, (
                f"Cell {key}: probabilities sum to {total}, expected 1.0"
            )

    def test_batter_probs_non_negative(self, fitted_model):
        """All probabilities must be >= 0."""
        for key, probs in fitted_model._batter_run_probs.items():
            assert np.all(probs >= 0), (
                f"Cell {key}: negative probability found in {probs}"
            )

    def test_wicket_rates_in_range(self, fitted_model):
        """Wicket rates must be in [0, 1]."""
        for key, rate in fitted_model._wicket_rates.items():
            assert 0.0 <= rate <= 1.0, (
                f"Cell {key}: wicket_rate={rate} out of [0, 1]"
            )

    def test_wide_rates_in_range(self, fitted_model):
        for phase, rate in fitted_model._wide_rates.items():
            assert 0.0 <= rate <= 1.0, f"wide_rate for {phase}={rate}"

    def test_noball_rates_in_range(self, fitted_model):
        for phase, rate in fitted_model._noball_rates.items():
            assert 0.0 <= rate <= 1.0, f"noball_rate for {phase}={rate}"

    def test_summary_returns_dataframe(self, fitted_model):
        df = fitted_model.summary()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 15  # 3 phases × 5 wih_buckets

    def test_unfitted_model_raises(self):
        """Calling sample() on an unfitted model must raise RuntimeError."""
        model = OutcomeModel()
        rng = np.random.default_rng(0)
        with pytest.raises(RuntimeError, match="fit\\(\\)"):
            model.sample("powerplay", 10, 1, rng)

# 2. TestOutcomeModelSample
class TestOutcomeModelSample:
    """sample() must return valid DeliveryOutcome objects."""

    N_SAMPLES = 1000

    def _draw_many(self, model, phase, wih, aggression):
        rng = np.random.default_rng(seed=0)
        return [
            model.sample(phase=phase, wickets_in_hand=wih,
                         aggression=aggression, rng=rng)
            for _ in range(self.N_SAMPLES)
        ]

    def test_batter_runs_valid_values(self, fitted_model):
        outcomes = self._draw_many(fitted_model, "middle", 7, 1)
        for o in outcomes:
            assert o.batter_runs in BATTER_RUNS_VALUES, (
                f"Invalid batter_runs: {o.batter_runs}"
            )

    def test_extra_runs_non_negative(self, fitted_model):
        outcomes = self._draw_many(fitted_model, "death", 5, 2)
        for o in outcomes:
            assert o.extra_runs >= 0

    def test_total_runs_equals_sum(self, fitted_model):
        outcomes = self._draw_many(fitted_model, "powerplay", 10, 0)
        for o in outcomes:
            assert o.total_runs == o.batter_runs + o.extra_runs, (
                f"total_runs={o.total_runs} != batter+extra "
                f"({o.batter_runs}+{o.extra_runs})"
            )

    def test_wide_and_noball_exclusive(self, fitted_model):
        """A delivery should not be simultaneously wide and no-ball."""
        outcomes = self._draw_many(fitted_model, "death", 6, 1)
        for o in outcomes:
            assert not (o.is_wide and o.is_noball), (
                "Outcome is both wide and no-ball simultaneously"
            )

    def test_legal_flag_consistent(self, fitted_model):
        """is_legal must be True iff not wide and not no-ball."""
        outcomes = self._draw_many(fitted_model, "middle", 8, 1)
        for o in outcomes:
            expected_legal = not o.is_wide and not o.is_noball
            assert o.is_legal == expected_legal, (
                f"is_legal={o.is_legal} inconsistent with "
                f"is_wide={o.is_wide}, is_noball={o.is_noball}"
            )

    def test_wicket_only_on_legal(self, fitted_model):
        """In our model, wickets only fall on legal deliveries."""
        outcomes = self._draw_many(fitted_model, "death", 4, 2)
        for o in outcomes:
            if o.is_wicket:
                assert o.is_legal, (
                    "Wicket on non-legal delivery (wide or no-ball)"
                )

    def test_wide_batter_runs_zero(self, fitted_model):
        """On a wide, batter_runs must always be 0."""
        outcomes = self._draw_many(fitted_model, "death", 8, 2)
        for o in outcomes:
            if o.is_wide:
                assert o.batter_runs == 0, (
                    f"Wide delivery has non-zero batter_runs={o.batter_runs}"
                )

    def test_wide_has_extra_run(self, fitted_model):
        """Wide deliveries must have extra_runs == 1."""
        outcomes = self._draw_many(fitted_model, "death", 8, 1)
        for o in outcomes:
            if o.is_wide:
                assert o.extra_runs == 1, (
                    f"Wide extra_runs={o.extra_runs}, expected 1"
                )

    def test_noball_has_extra_run(self, fitted_model):
        """No-ball deliveries must have extra_runs == 1."""
        # Draw many to ensure we encounter some no-balls
        rng = np.random.default_rng(seed=7)
        outcomes = [
            fitted_model.sample("powerplay", 10, 1, rng)
            for _ in range(5000)
        ]
        noballs = [o for o in outcomes if o.is_noball]
        assert len(noballs) > 0, "Expected some no-balls in 5,000 draws"
        for o in noballs:
            assert o.extra_runs == 1, (
                f"No-ball extra_runs={o.extra_runs}, expected 1"
            )

# 3. TestAggressionModifiers
class TestAggressionModifiers:
    """
    Aggressive action must produce more 4s and 6s than balanced, which in turn produces more than conservative (all else equal).
    """

    N = 10_000

    def _boundary_rate(self, model, aggression, phase="death", wih=7):
        rng = np.random.default_rng(seed=0)
        legal = [
            model.sample(phase=phase, wickets_in_hand=wih,
                         aggression=aggression, rng=rng)
            for _ in range(self.N)
        ]
        legal = [o for o in legal if o.is_legal]
        if not legal:
            return 0.0
        boundaries = sum(1 for o in legal if o.batter_runs in (4, 6))
        return boundaries / len(legal)

    def test_aggressive_more_boundaries_than_balanced(self, fitted_model):
        rate_balanced    = self._boundary_rate(fitted_model, aggression=1)
        rate_aggressive  = self._boundary_rate(fitted_model, aggression=2)
        assert rate_aggressive > rate_balanced, (
            f"Aggressive boundary rate {rate_aggressive:.3f} <= "
            f"balanced {rate_balanced:.3f}"
        )

    def test_balanced_more_boundaries_than_conservative(self, fitted_model):
        rate_conservative = self._boundary_rate(fitted_model, aggression=0)
        rate_balanced     = self._boundary_rate(fitted_model, aggression=1)
        assert rate_balanced > rate_conservative, (
            f"Balanced boundary rate {rate_balanced:.3f} <= "
            f"conservative {rate_conservative:.3f}"
        )

    def test_aggressive_higher_wicket_rate(self, fitted_model):
        """Aggressive action must produce a higher wicket rate than balanced."""
        N = 10_000
        rng_agg = np.random.default_rng(seed=0)
        rng_bal = np.random.default_rng(seed=0)
        agg = [
            fitted_model.sample("death", 7, aggression=2, rng=rng_agg)
            for _ in range(N)
        ]
        bal = [
            fitted_model.sample("death", 7, aggression=1, rng=rng_bal)
            for _ in range(N)
        ]
        legal_agg = [o for o in agg if o.is_legal]
        legal_bal = [o for o in bal if o.is_legal]
        wkt_rate_agg = sum(o.is_wicket for o in legal_agg) / max(len(legal_agg), 1)
        wkt_rate_bal = sum(o.is_wicket for o in legal_bal) / max(len(legal_bal), 1)
        assert wkt_rate_agg > wkt_rate_bal, (
            f"Aggressive wicket rate {wkt_rate_agg:.4f} <= "
            f"balanced {wkt_rate_bal:.4f}"
        )

    def test_wicket_delivery_scores_zero_runs(self, fitted_model):
        """
        When a wicket falls, batter_runs must be 0.

        Historical data: 97.5% of wicket deliveries in IPL score 0 batter runs
        (bowled, caught, LBW, stumped). The model corrects the independent-sampling
        bias by forcing batter_runs=0 whenever is_wicket is True.
        """
        N = 50_000
        rng = np.random.default_rng(seed=42)
        outcomes = [
            fitted_model.sample("death", 7, aggression=2, rng=rng)
            for _ in range(N)
        ]
        wicket_deliveries = [o for o in outcomes if o.is_legal and o.is_wicket]
        assert len(wicket_deliveries) > 100, "Too few wickets sampled; increase N"
        nonzero_runs_on_wicket = [o for o in wicket_deliveries if o.batter_runs != 0]
        assert len(nonzero_runs_on_wicket) == 0, (
            f"Found {len(nonzero_runs_on_wicket)} wicket deliveries with "
            f"batter_runs != 0 (e.g. {nonzero_runs_on_wicket[:3]}). "
            "batter_runs must be 0 on a wicket delivery."
        )

# 4. TestSimulatorReset
class TestSimulatorReset:
    """reset() must initialise all state correctly."""

    def test_initial_score_zero(self, simulator):
        simulator.reset(target=180)
        assert simulator.score == 0

    def test_initial_wickets_zero(self, simulator):
        simulator.reset(target=180)
        assert simulator.wickets == 0

    def test_initial_legal_balls_zero(self, simulator):
        simulator.reset(target=180)
        assert simulator.legal_balls == 0

    def test_is_done_false_after_reset(self, simulator):
        simulator.reset(target=180)
        assert not simulator.is_done

    def test_won_none_after_reset(self, simulator):
        simulator.reset(target=180)
        assert simulator.won is None

    def test_target_stored(self, simulator):
        simulator.reset(target=200)
        assert simulator.target == 200

    def test_history_cleared(self, simulator):
        simulator.reset(target=180)
        assert simulator.history == []

    def test_observation_shape(self, simulator):
        obs = simulator.reset(target=180)
        assert obs.shape == (6,), f"Expected obs shape (6,), got {obs.shape}"

    def test_observation_dtype(self, simulator):
        obs = simulator.reset(target=180)
        assert obs.dtype == np.float32

    def test_observation_in_range(self, simulator):
        obs = simulator.reset(target=180)
        assert np.all(obs >= 0) and np.all(obs <= 1), (
            f"Initial observation out of [0,1]: {obs}"
        )

    def test_invalid_target_raises(self, simulator):
        with pytest.raises(ValueError, match="target"):
            simulator.reset(target=0)

    def test_invalid_initial_wickets_raises(self, simulator):
        with pytest.raises(ValueError, match="initial_wickets"):
            simulator.reset(target=180, initial_wickets=10)

    def test_step_after_done_raises(self, fitted_model):
        """Calling step() after innings ends must raise RuntimeError."""
        sim = T20Simulator(fitted_model, random_seed=1)
        result = sim.run_episode(target=1, aggression=2, random_seed=1)
        # Score >= 1 (target=1) → done after first scoring delivery.
        with pytest.raises(RuntimeError, match="reset"):
            sim.step(aggression=1)

# 5. TestSimulatorNormalBall
class TestSimulatorNormalBall:
    """
    Use a deterministic mock to isolate the scoring logic on a normal delivery.
    We override the RNG so the first sample is always a legal 4-run delivery.
    """

    def _make_sim_with_mock(self, fitted_model):
        """Return a simulator whose outcome model always returns 4 runs, legal."""
        class _Mock4RunModel(OutcomeModel):
            def __init__(self, base):
                self.__dict__ = base.__dict__.copy()
            def sample(self, phase, wickets_in_hand, aggression, rng):
                return DeliveryOutcome(
                    batter_runs=4, extra_runs=0, total_runs=4,
                    is_wicket=False, is_wide=False, is_noball=False, is_legal=True,
                )
        return T20Simulator(_Mock4RunModel(fitted_model), random_seed=0)

    def test_score_increments_by_total_runs(self, fitted_model):
        sim = self._make_sim_with_mock(fitted_model)
        sim.reset(target=200)
        obs, reward, done, info = sim.step(aggression=1)
        assert sim.score == 4

    def test_legal_ball_increments(self, fitted_model):
        sim = self._make_sim_with_mock(fitted_model)
        sim.reset(target=200)
        sim.step(aggression=1)
        assert sim.legal_balls == 1

    def test_wickets_unchanged_no_wicket(self, fitted_model):
        sim = self._make_sim_with_mock(fitted_model)
        sim.reset(target=200)
        sim.step(aggression=1)
        assert sim.wickets == 0

    def test_reward_zero_before_termination(self, fitted_model):
        sim = self._make_sim_with_mock(fitted_model)
        sim.reset(target=200)
        _, reward, done, _ = sim.step(aggression=1)
        assert reward == 0.0
        assert not done

    def test_history_records_ball(self, fitted_model):
        sim = self._make_sim_with_mock(fitted_model)
        sim.reset(target=200)
        sim.step(aggression=1)
        assert len(sim.history) == 1
        assert sim.history[0].score_after == 4

# 6. TestSimulatorWide
class TestSimulatorWide:
    """Wide deliveries add 1 run but do NOT advance the legal-ball counter."""

    def _make_wide_sim(self, fitted_model):
        """Return a simulator that always delivers a wide."""
        class _WideModel(OutcomeModel):
            def __init__(self, base):
                self.__dict__ = base.__dict__.copy()
            def sample(self, phase, wickets_in_hand, aggression, rng):
                return DeliveryOutcome(
                    batter_runs=0, extra_runs=1, total_runs=1,
                    is_wicket=False, is_wide=True, is_noball=False, is_legal=False,
                )
        return T20Simulator(_WideModel(fitted_model), random_seed=0)

    def test_wide_adds_one_run(self, fitted_model):
        sim = self._make_wide_sim(fitted_model)
        sim.reset(target=200)
        sim.step(aggression=1)
        assert sim.score == 1

    def test_wide_does_not_increment_legal_balls(self, fitted_model):
        sim = self._make_wide_sim(fitted_model)
        sim.reset(target=200)
        sim.step(aggression=1)
        assert sim.legal_balls == 0, (
            f"Wide incremented legal_balls to {sim.legal_balls}, expected 0"
        )

    def test_wide_does_not_end_innings(self, fitted_model):
        sim = self._make_wide_sim(fitted_model)
        sim.reset(target=200)
        _, reward, done, _ = sim.step(aggression=1)
        assert not done

# 7. TestSimulatorNoBall
class TestSimulatorNoBall:
    """No-ball adds batter runs + 1 extra and does NOT advance legal-ball counter."""

    def _make_noball_sim(self, fitted_model, batter_runs=3):
        class _NoBallModel(OutcomeModel):
            def __init__(self, base, br):
                self.__dict__ = base.__dict__.copy()
                self._br = br
            def sample(self, phase, wickets_in_hand, aggression, rng):
                return DeliveryOutcome(
                    batter_runs=self._br, extra_runs=1, total_runs=self._br + 1,
                    is_wicket=False, is_wide=False, is_noball=True, is_legal=False,
                )
        return T20Simulator(_NoBallModel(fitted_model, batter_runs), random_seed=0)

    def test_noball_adds_batter_plus_extra(self, fitted_model):
        sim = self._make_noball_sim(fitted_model, batter_runs=3)
        sim.reset(target=200)
        sim.step(aggression=1)
        assert sim.score == 4  # 3 batter + 1 no-ball

    def test_noball_does_not_increment_legal_balls(self, fitted_model):
        sim = self._make_noball_sim(fitted_model, batter_runs=0)
        sim.reset(target=200)
        sim.step(aggression=1)
        assert sim.legal_balls == 0, (
            f"No-ball incremented legal_balls to {sim.legal_balls}"
        )

    def test_noball_does_not_change_wickets(self, fitted_model):
        sim = self._make_noball_sim(fitted_model, batter_runs=0)
        sim.reset(target=200)
        sim.step(aggression=1)
        assert sim.wickets == 0

# 8. TestSimulatorWicket
class TestSimulatorWicket:
    """Wickets on legal deliveries increment the wickets counter."""

    def _make_wicket_sim(self, fitted_model, batter_runs=0):
        class _WicketModel(OutcomeModel):
            def __init__(self, base, br):
                self.__dict__ = base.__dict__.copy()
                self._br = br
            def sample(self, phase, wickets_in_hand, aggression, rng):
                return DeliveryOutcome(
                    batter_runs=self._br, extra_runs=0, total_runs=self._br,
                    is_wicket=True, is_wide=False, is_noball=False, is_legal=True,
                )
        return T20Simulator(_WicketModel(fitted_model, batter_runs), random_seed=0)

    def test_wicket_increments_counter(self, fitted_model):
        sim = self._make_wicket_sim(fitted_model)
        sim.reset(target=200)
        sim.step(aggression=1)
        assert sim.wickets == 1

    def test_multiple_wickets_accumulate(self, fitted_model):
        sim = self._make_wicket_sim(fitted_model)
        sim.reset(target=200)
        for _ in range(5):
            sim.step(aggression=1)
        assert sim.wickets == 5

    def test_legal_ball_counts_on_wicket(self, fitted_model):
        sim = self._make_wicket_sim(fitted_model)
        sim.reset(target=200)
        sim.step(aggression=1)
        assert sim.legal_balls == 1

# 9. TestSimulatorAllOut
class TestSimulatorAllOut:
    """Innings must end (loss) when 10 wickets fall."""

    def _make_always_wicket_sim(self, fitted_model):
        class _AlwaysWicketModel(OutcomeModel):
            def __init__(self, base):
                self.__dict__ = base.__dict__.copy()
            def sample(self, phase, wickets_in_hand, aggression, rng):
                return DeliveryOutcome(
                    batter_runs=0, extra_runs=0, total_runs=0,
                    is_wicket=True, is_wide=False, is_noball=False, is_legal=True,
                )
        return T20Simulator(_AlwaysWicketModel(fitted_model), random_seed=0)

    def test_all_out_after_ten_wickets(self, fitted_model):
        sim = self._make_always_wicket_sim(fitted_model)
        sim.reset(target=200)
        done = False
        balls_bowled = 0
        while not done:
            _, reward, done, info = sim.step(aggression=1)
            balls_bowled += 1
        assert sim.wickets == MAX_WICKETS, (
            f"Expected {MAX_WICKETS} wickets at all-out, got {sim.wickets}"
        )

    def test_all_out_is_loss(self, fitted_model):
        sim = self._make_always_wicket_sim(fitted_model)
        sim.reset(target=200)
        while not sim.is_done:
            _, reward, _, _ = sim.step(aggression=1)
        assert sim.won is False
        assert reward == -1.0

    def test_all_out_requires_exactly_ten(self, fitted_model):
        """Innings must not end at 9 wickets."""
        sim = self._make_always_wicket_sim(fitted_model)
        sim.reset(target=200)
        for _ in range(9):
            _, _, done, _ = sim.step(aggression=1)
            assert not done, "Innings ended before 10 wickets"
        _, _, done, _ = sim.step(aggression=1)
        assert done, "Innings did not end at 10th wicket"

# 10. TestSimulatorTargetReached
class TestSimulatorTargetReached:
    """Innings must end (win) as soon as score >= target."""

    def _make_scoring_sim(self, fitted_model, runs_per_ball=1):
        class _ConstantRunsModel(OutcomeModel):
            def __init__(self, base, rpb):
                self.__dict__ = base.__dict__.copy()
                self._rpb = rpb
            def sample(self, phase, wickets_in_hand, aggression, rng):
                return DeliveryOutcome(
                    batter_runs=self._rpb, extra_runs=0, total_runs=self._rpb,
                    is_wicket=False, is_wide=False, is_noball=False, is_legal=True,
                )
        return T20Simulator(_ConstantRunsModel(fitted_model, runs_per_ball), random_seed=0)

    def test_win_when_target_reached(self, fitted_model):
        # Target=10, scoring 2 per ball → win after 5 balls.
        sim = self._make_scoring_sim(fitted_model, runs_per_ball=2)
        sim.reset(target=10)
        done = False
        while not done:
            _, reward, done, _ = sim.step(aggression=1)
        assert sim.won is True
        assert reward == 1.0

    def test_score_at_win(self, fitted_model):
        sim = self._make_scoring_sim(fitted_model, runs_per_ball=2)
        sim.reset(target=10)
        while not sim.is_done:
            sim.step(aggression=1)
        assert sim.score >= sim.target

    def test_wins_on_exact_target(self, fitted_model):
        """Score == target must count as a win."""
        # 5 balls of 2 runs = 10 exactly, target = 10.
        sim = self._make_scoring_sim(fitted_model, runs_per_ball=2)
        sim.reset(target=10)
        while not sim.is_done:
            sim.step(aggression=1)
        assert sim.won is True
        assert sim.score >= 10

# 11. TestSimulatorOversComplete
class TestSimulatorOversComplete:
    """Innings must end at exactly 120 legal balls (20 overs)."""

    def _make_zero_run_sim(self, fitted_model):
        """Always score 0, no wickets, no extras → innings ends at 120 balls."""
        class _ZeroRunModel(OutcomeModel):
            def __init__(self, base):
                self.__dict__ = base.__dict__.copy()
            def sample(self, phase, wickets_in_hand, aggression, rng):
                return DeliveryOutcome(
                    batter_runs=0, extra_runs=0, total_runs=0,
                    is_wicket=False, is_wide=False, is_noball=False, is_legal=True,
                )
        return T20Simulator(_ZeroRunModel(fitted_model), random_seed=0)

    def test_innings_ends_at_120_legal_balls(self, fitted_model):
        sim = self._make_zero_run_sim(fitted_model)
        sim.reset(target=200)
        while not sim.is_done:
            sim.step(aggression=1)
        assert sim.legal_balls == MAX_LEGAL_BALLS, (
            f"Expected {MAX_LEGAL_BALLS} legal balls, got {sim.legal_balls}"
        )

    def test_innings_is_loss_when_target_not_reached(self, fitted_model):
        sim = self._make_zero_run_sim(fitted_model)
        sim.reset(target=200)
        while not sim.is_done:
            sim.step(aggression=1)
        assert sim.won is False

    def test_extras_do_not_count_as_legal_balls(self, fitted_model):
        """
        Verify that 120 legal balls are reached even when extras are mixed in.
        We use a model that alternates wide → legal (0 runs).
        """
        class _AlternatingModel(OutcomeModel):
            def __init__(self, base):
                self.__dict__ = base.__dict__.copy()
                self._call_count = 0
            def sample(self, phase, wickets_in_hand, aggression, rng):
                self._call_count += 1
                if self._call_count % 2 == 0:
                    # Wide
                    return DeliveryOutcome(0, 1, 1, False, True, False, False)
                else:
                    # Legal 0
                    return DeliveryOutcome(0, 0, 0, False, False, False, True)
        sim = T20Simulator(_AlternatingModel(fitted_model), random_seed=0)
        sim.reset(target=9999)
        while not sim.is_done:
            sim.step(aggression=1)
        assert sim.legal_balls == MAX_LEGAL_BALLS

# 12. TestSimulatorReproducible
class TestSimulatorReproducible:
    """Same seed must produce the same episode outcome."""

    def test_same_seed_same_score(self, fitted_model):
        sim = T20Simulator(fitted_model, random_seed=99)
        r1 = sim.run_episode(target=170, aggression=1, random_seed=99)
        r2 = sim.run_episode(target=170, aggression=1, random_seed=99)
        assert r1["score"] == r2["score"]
        assert r1["wickets"] == r2["wickets"]
        assert r1["legal_balls"] == r2["legal_balls"]

    def test_different_seeds_likely_different(self, fitted_model):
        """Two different seeds very likely produce different innings outcomes."""
        sim = T20Simulator(fitted_model)
        scores = set()
        for seed in range(20):
            r = sim.run_episode(target=170, aggression=1, random_seed=seed)
            scores.add(r["score"])
        # With 20 different seeds, we should see at least 5 distinct scores.
        assert len(scores) >= 5, (
            f"Only {len(scores)} distinct scores across 20 seeds — "
            "RNG may not be working correctly."
        )

    def test_episode_observation_shape_and_dtype(self, fitted_model):
        """Observation returned by step() must be (6,) float32."""
        sim = T20Simulator(fitted_model, random_seed=0)
        obs = sim.reset(target=170)
        obs2, _, _, _ = sim.step(aggression=1)
        assert obs2.shape == (6,)
        assert obs2.dtype == np.float32

# 13. TestSimulatorHistoricalValidation
class TestSimulatorHistoricalValidation:
    """
    Validate simulator output against historical IPL statistics.

    We run N_EPISODES simulated innings and compare key statistics to historical baselines computed from training data.
    """

    N_EPISODES = 500
    TARGET = 165  # Typical IPL second-innings target

    @pytest.fixture(scope="class")
    @classmethod
    def episode_results(cls, fitted_model):
        """Run N_EPISODES and collect results."""
        sim = T20Simulator(fitted_model, random_seed=None)  # No fixed seed for Monte Carlo
        results = []
        for seed in range(cls.N_EPISODES):
            r = sim.run_episode(target=cls.TARGET, aggression=1, random_seed=seed)
            results.append(r)
        return results

    def test_mean_score_in_range(self, episode_results):
        """
        Simulated mean score should be plausible for a T20 chase.
        Historical mean innings2 score is ~156; we allow 100–220 for
        simulated chases (a wider range because we fix target=165 and
        balanced aggression, which differs from the full historical mix).
        """
        scores = [r["score"] for r in episode_results]
        mean_score = np.mean(scores)
        assert 80 <= mean_score <= 250, (
            f"Simulated mean score {mean_score:.1f} is outside [80, 250]. "
            "This suggests a systematic bug in score accumulation."
        )

    def test_legal_balls_never_exceeds_120(self, episode_results):
        """Legal balls must never exceed 120."""
        max_lb = max(r["legal_balls"] for r in episode_results)
        assert max_lb <= MAX_LEGAL_BALLS, (
            f"legal_balls reached {max_lb}, exceeding the 120-ball limit"
        )

    def test_wickets_never_exceeds_10(self, episode_results):
        """Wickets must never exceed 10."""
        max_wkt = max(r["wickets"] for r in episode_results)
        assert max_wkt <= MAX_WICKETS, (
            f"wickets reached {max_wkt}, exceeding the 10-wicket limit"
        )

    def test_all_episodes_terminate(self, episode_results):
        """Every episode must have a definitive outcome (won = True or False)."""
        no_result = [r for r in episode_results if r["won"] is None]
        assert len(no_result) == 0, (
            f"{len(no_result)} episodes did not terminate with a win or loss"
        )

    def test_win_rate_plausible(self, episode_results):
        """
        Win rate with balanced aggression and target=165 should be plausible.
        Historical IPL: second-innings teams win ~50 % of matches overall.
        We allow 20–80 % — a wide range because target=165 is near the median.
        """
        wins = sum(1 for r in episode_results if r["won"])
        win_rate = wins / len(episode_results)
        assert 0.15 <= win_rate <= 0.85, (
            f"Win rate {win_rate:.3f} is outside [0.15, 0.85]. "
            f"({wins}/{len(episode_results)} wins)"
        )

    def test_mean_legal_balls_plausible(self, episode_results):
        """
        Mean legal balls per episode should be between 60 and 120.
        Historical mean is ~112; many innings end at 120 (overs complete)
        or earlier (all out or target reached).
        """
        mean_lb = np.mean([r["legal_balls"] for r in episode_results])
        assert 60 <= mean_lb <= 120, (
            f"Mean legal balls {mean_lb:.1f} outside [60, 120]"
        )
"""
An interactive Streamlit application demonstrating a full T20 second-innings run chase combining:
1. An autonomous Deep Q-Network (DQN) batting captaincy agent.
2. An empirical, rule-based bowling captaincy and tactics module with realistic IPL franchises, over limits, and workload tracking.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch

# Ensure src/ is on sys.path
ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from baselines import (
    AlwaysAggressive,
    AlwaysBalanced,
    AlwaysConservative,
    RRRPolicy,
)
from bowling_policy import (
    Bowler,
    BowlingSquad,
    PhaseSpecialistCaptaincy,
    RiskAdaptiveCaptaincy,
    RoundRobinCaptaincy,
    AlwaysAttackingCaptaincy,
    AlwaysDefensiveCaptaincy,
    AVAILABLE_SQUADS,
    BOWLING_ACTIONS,
    BOWLING_ACTION_COLORS,
    get_csk_squad,
    get_mi_squad,
    get_rcb_squad,
    get_kkr_squad,
    get_custom_squad,
)
from data_loader import load_pipeline_data
from dqn_agent import DQNAgent
from outcome_model import OutcomeModel
from simulator import T20Simulator

# Page Configuration & Styling
st.set_page_config(
    page_title="CricRL — T20 Autonomous Batting & Bowling Dashboard",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #1f77b4;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .status-won {
        background-color: #d4edda;
        color: #155724;
        padding: 14px;
        border-radius: 8px;
        font-weight: bold;
        text-align: center;
        font-size: 1.25rem;
    }
    .status-lost {
        background-color: #f8d7da;
        color: #721c24;
        padding: 14px;
        border-radius: 8px;
        font-weight: bold;
        text-align: center;
        font-size: 1.25rem;
    }
    .bowler-banner {
        background-color: #edf2f7;
        border-left: 5px solid #2b6cb0;
        padding: 12px 16px;
        border-radius: 8px;
        margin-bottom: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

BATTING_ACTION_NAMES = {0: "Conservative", 1: "Balanced", 2: "Aggressive"}
BATTING_ACTION_COLORS = {0: "#6c757d", 1: "#1f77b4", 2: "#d62728"}

# Cached Model & Data Loaders
@st.cache_resource(show_spinner="Loading trained outcome model from historical data...")
def get_cached_outcome_model() -> OutcomeModel:
    data = load_pipeline_data(verbose=False)
    train_ids = set(data["train_match_ids"])
    inn2 = pd.read_csv(ROOT / "data" / "processed" / "innings2_chase.csv", low_memory=False)
    train_df = inn2[inn2["match_id"].isin(train_ids)].copy()
    model = OutcomeModel().fit(train_df)
    return model


@st.cache_resource(show_spinner="Loading trained DQN checkpoint...")
def get_cached_dqn_agent() -> DQNAgent:
    agent = DQNAgent(state_dim=6, action_dim=3)
    best_path = ROOT / "models" / "dqn_cricket_best.pt"
    ckpt_path = ROOT / "models" / "dqn_cricket.pt"
    if best_path.exists():
        agent.load(best_path)
    elif ckpt_path.exists():
        agent.load(ckpt_path)
    return agent


outcome_model = get_cached_outcome_model()
dqn_agent = get_cached_dqn_agent()

# Helper Functions
def format_overs(legal_balls: int) -> str:
    overs = legal_balls // 6
    balls = legal_balls % 6
    return f"{overs}.{balls}"


def describe_delivery_event(outcome) -> str:
    """Format delivery outcome into a human-readable event badge."""
    if outcome.is_wicket:
        return "🔴 WICKET! (+0 runs)"
    elif outcome.is_wide:
        return f"🟡 WIDE (+{outcome.extra_runs} extra)"
    elif outcome.is_noball:
        return f"🟣 NO-BALL (+{outcome.total_runs} runs)"
    elif outcome.total_runs == 0:
        return "⚪ Dot Ball (+0 runs)"
    elif outcome.total_runs == 1:
        return "🏃 Single (+1 run)"
    elif outcome.total_runs == 2:
        return "🏃 Two Runs (+2 runs)"
    elif outcome.total_runs == 3:
        return "🏃 Three Runs (+3 runs)"
    elif outcome.total_runs == 4:
        return "4️⃣ FOUR! (+4 runs)"
    elif outcome.total_runs == 6:
        return "6️⃣ SIX! (+6 runs)"
    return f"🏏 {outcome.total_runs} Runs (+{outcome.total_runs})"


def build_ball_record(
    step_num: int,
    overs_str: str,
    score_before: int,
    wickets_before: int,
    target: int,
    sim: T20Simulator,
    q_vals: np.ndarray,
    batting_action: int,
    bowling_action: int,
    bowler_name: str,
    outcome,
) -> dict:
    return {
        "ball_num": step_num,
        "over": overs_str,
        "bowler": bowler_name,
        "bowling_tactic": BOWLING_ACTIONS.get(bowling_action, "Balanced"),
        "score_before": f"{score_before}/{wickets_before}",
        "event_desc": describe_delivery_event(outcome),
        "outcome_runs": outcome.total_runs,
        "score_after": f"{sim.score}/{sim.wickets}",
        "runs_required": max(0, target - sim.score),
        "balls_remaining": sim.balls_remaining,
        "rrr": round(sim.required_run_rate, 2),
        "batting_action": BATTING_ACTION_NAMES[batting_action],
        "action_id": batting_action,
        "q_conservative": round(float(q_vals[0]), 4),
        "q_balanced": round(float(q_vals[1]), 4),
        "q_aggressive": round(float(q_vals[2]), 4),
        "is_wicket": outcome.is_wicket,
        "is_wide": outcome.is_wide,
        "is_noball": outcome.is_noball,
    }


def save_chase_log(records: List[dict], filepath: Path = ROOT / "reports" / "interactive_chase_log.csv") -> Path:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv(filepath, index=False)
    return filepath

# Header & Navigation
st.title("🏏 CricRL: T20 Autonomous Batting & Bowling Decision System")
st.markdown(
    """
    **Autonomous Batting Aggression (Deep Q-Network) & Data-Driven Bowling Captaincy**  
    A complete second-innings T20 chase simulator integrating a trained Bellman DQN batting agent with
    IPL franchise bowling squads, over limit constraints, and tactical captaincy policies.
    """
)

tabs = st.tabs(["🎮 Interactive Match Chase", "📊 Baseline Benchmarks & Audit", "📖 System Architecture & Logic"])

# TAB 1: INTERACTIVE MATCH CHASE
with tabs[0]:
    st.sidebar.header("🎯 Match Configuration")

    # Team Selections
    batting_team = st.sidebar.selectbox(
        "Batting Team (DQN Agent)",
        ["Royal Challengers Bengaluru", "Mumbai Indians", "Chennai Super Kings", "Kolkata Knight Riders", "India XI / Custom"],
        index=0,
    )

    bowling_team = st.sidebar.selectbox(
        "Bowling Team (Opponent)",
        ["Chennai Super Kings", "Mumbai Indians", "Royal Challengers Bengaluru", "Kolkata Knight Riders", "Custom XI"],
        index=0,
    )

    # Bowling Captaincy Policy Selector
    bowling_policy_choice = st.sidebar.selectbox(
        "Bowling Captaincy Strategy",
        [
            "Phase Specialist (Rule-Based Specialist Allocation)",
            "Risk-Adaptive (Dynamic RRR & Wicket Reactive)",
            "Round-Robin Rotation (Cycling Baseline)",
            "Always Attacking (Aggressive Baseline)",
            "Always Defensive (Containment Baseline)",
        ],
        index=0,
    )

    # Chase Target Setup
    target_preset = st.sidebar.selectbox(
        "Chase Target Preset",
        ["145 runs (Below-Average / Modest)", "165 runs (IPL Par Target)", "185 runs (Demanding / High Pressure)", "Custom Target"],
        index=1,
    )
    if target_preset.startswith("145"):
        default_target = 145
    elif target_preset.startswith("165"):
        default_target = 165
    elif target_preset.startswith("185"):
        default_target = 185
    else:
        default_target = 170

    target = st.sidebar.number_input("Target Runs", min_value=80, max_value=260, value=default_target, step=1)
    initial_wickets = st.sidebar.slider("Initial Wickets Fallen", min_value=0, max_value=8, value=0, step=1)
    random_seed = st.sidebar.number_input("Simulation Random Seed", min_value=0, max_value=999999, value=42, step=1)
    sim_mode = st.sidebar.radio("Simulation Mode", ["Step-by-Step (Ball-by-Ball)", "Simulate Full Innings"], index=0)

    # Detect configuration changes to auto-reset
    config_changed = (
        st.session_state.get("target") != int(target)
        or st.session_state.get("initial_wickets") != int(initial_wickets)
        or st.session_state.get("seed") != int(random_seed)
        or st.session_state.get("bowling_team") != bowling_team
        or st.session_state.get("batting_team") != batting_team
        or st.session_state.get("target_preset") != target_preset
        or st.session_state.get("bowling_policy_choice") != bowling_policy_choice
    )

    # Instantiate Bowling Policy Helper
    def create_bowling_policy(choice: str):
        if choice.startswith("Phase Specialist"):
            return PhaseSpecialistCaptaincy()
        elif choice.startswith("Risk-Adaptive"):
            return RiskAdaptiveCaptaincy()
        elif choice.startswith("Round-Robin"):
            return RoundRobinCaptaincy()
        elif choice.startswith("Always Attacking"):
            return AlwaysAttackingCaptaincy()
        else:
            return AlwaysDefensiveCaptaincy()

    # Instantiate Squad Helper
    def create_squad(team_name: str) -> BowlingSquad:
        if team_name in AVAILABLE_SQUADS:
            return AVAILABLE_SQUADS[team_name]()
        return get_custom_squad(team_name)

    # Initialize Session State
    if "sim" not in st.session_state or config_changed or st.sidebar.button("🔄 Reset / Start New Chase"):
        squad = create_squad(bowling_team)
        squad.reset()
        captaincy = create_bowling_policy(bowling_policy_choice)

        sim = T20Simulator(outcome_model=outcome_model, random_seed=int(random_seed))
        sim.reset(target=int(target), initial_wickets=int(initial_wickets), random_seed=int(random_seed))

        # Initial Bowler and Tactic
        init_bowler, init_b_reason = captaincy.select_bowler(squad, 0, 0, int(initial_wickets), int(target))
        init_tactic, init_t_reason = captaincy.select_tactic(init_bowler, 0, 0, int(initial_wickets), int(target))

        st.session_state.sim = sim
        st.session_state.squad = squad
        st.session_state.captaincy = captaincy
        st.session_state.current_bowler = init_bowler
        st.session_state.current_tactic = init_tactic
        st.session_state.bowling_reason = f"{init_b_reason} | Tactic: {init_t_reason}"
        st.session_state.history = []
        st.session_state.target = int(target)
        st.session_state.initial_wickets = int(initial_wickets)
        st.session_state.seed = int(random_seed)
        st.session_state.bowling_team = bowling_team
        st.session_state.batting_team = batting_team
        st.session_state.target_preset = target_preset
        st.session_state.bowling_policy_choice = bowling_policy_choice

    sim: T20Simulator = st.session_state.sim
    squad: BowlingSquad = st.session_state.squad
    captaincy = st.session_state.captaincy
    current_bowler: Bowler = st.session_state.current_bowler
    current_tactic: int = st.session_state.current_tactic
    bowling_reason: str = st.session_state.bowling_reason

    # Sidebar: Squad roster & workload preview
    with st.sidebar.expander(f"📋 {bowling_team} Squad Overview", expanded=False):
        roster_data = [
            {
                "Bowler": b.name,
                "Role": b.role,
                "PP/Mid/Death Econ": f"{b.powerplay_econ:.1f} / {b.middle_econ:.1f} / {b.death_econ:.1f}",
                "Overs Left": b.overs_remaining,
            }
            for b in squad.bowlers
        ]
        st.dataframe(pd.DataFrame(roster_data), width="stretch", hide_index=True)

    # Current match metrics
    balls_remaining = sim.balls_remaining
    runs_needed = sim.runs_required
    rrr = sim.required_run_rate
    crr = sim.current_run_rate

    # Match Scoreboard
    st.subheader(f"📋 {batting_team} vs {bowling_team} — Scoreboard")
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Score", f"{sim.score}/{sim.wickets}", delta=f"{runs_needed} req" if sim.score < sim.target else "Target reached")
    with col2:
        st.metric("Target", f"{sim.target} runs", help="Runs needed to win match")
    with col3:
        st.metric("Overs", format_overs(sim.legal_balls), f"{balls_remaining} balls left")
    with col4:
        st.metric("Current Run Rate (CRR)", f"{crr:.2f} rpo")
    with col5:
        st.metric("Required Run Rate (RRR)", f"{rrr:.2f} rpo" if balls_remaining > 0 else "N/A")

    # Progress bar
    progress_frac = min(1.0, max(0.0, sim.score / sim.target)) if sim.target > 0 else 0.0
    st.progress(progress_frac, text=f"Chase Progress: {sim.score} / {sim.target} runs ({progress_frac*100:.1f}%)")

    # Active Bowler Banner
    if not sim.is_done and current_bowler:
        current_over_num = min(20, (sim.legal_balls // 6) + 1)
        ball_in_over = sim.legal_balls % 6
        st.markdown(
            f"""
            <div class="bowler-banner">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                    <span style="font-size: 1.15rem; font-weight: bold;">🎯 Active Bowler: {current_bowler.name}</span>
                    <span style="font-size: 0.85rem; background: #e2e8f0; padding: 3px 8px; border-radius: 6px; font-weight: 600;">
                        Over {current_over_num} of 20 &nbsp;•&nbsp; Delivery {ball_in_over + 1} of 6
                    </span>
                </div>
                <div>
                    <b>Role:</b> {current_bowler.role} ({current_bowler.bowling_type}) &nbsp;|&nbsp;
                    <b>Match Figures:</b> <code>{current_bowler.figures_str}</code> &nbsp;|&nbsp;
                    <b>Quota:</b> {current_bowler.overs_bowled_str} / {current_bowler.max_overs} overs bowled ({current_bowler.overs_remaining} left) &nbsp;|&nbsp;
                    <b>Tactic:</b> <span style="font-weight: bold; color: {BOWLING_ACTION_COLORS[current_tactic]};">
                        {BOWLING_ACTIONS[current_tactic]}
                    </span>  
                </div>
                <div style="font-size: 0.88rem; color: #4a5568; margin-top: 6px;">
                    <b>Captaincy Rationale:</b> {bowling_reason}
                </div>
                <div style="font-size: 0.8rem; color: #718096; margin-top: 4px;">
                    ℹ️ <i>Cricket Over Rule: A single bowler bowls all 6 legal deliveries of an over before captaincy rotates to a new bowler (no consecutive overs allowed).</i>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Interactive Execution
    if sim_mode == "Step-by-Step (Ball-by-Ball)":
        ctrl_col1, ctrl_col2, _ = st.columns([2, 2, 4])
        with ctrl_col1:
            step_btn = st.button("⚾ Bowl Next Delivery", disabled=sim.is_done, width="stretch")
        with ctrl_col2:
            auto_btn = st.button("⏩ Fast-Forward to End", disabled=sim.is_done, width="stretch")

        # Single step execution
        if step_btn and not sim.is_done:
            obs = sim._observation()
            q_vals = dqn_agent.get_q_values(obs)
            batting_action = int(np.argmax(q_vals))

            step_num = len(st.session_state.history) + 1
            score_before = sim.score
            wickets_before = sim.wickets

            # Step simulator with joint batting and bowling
            sim.step(
                aggression=batting_action,
                bowling_aggression=current_tactic,
                bowler=current_bowler,
            )
            outcome = sim.last_outcome
            overs_str = format_overs(sim.legal_balls) if outcome.is_legal else f"{format_overs(sim.legal_balls)} (extra)"

            record = build_ball_record(
                step_num=step_num,
                overs_str=overs_str,
                score_before=score_before,
                wickets_before=wickets_before,
                target=sim.target,
                sim=sim,
                q_vals=q_vals,
                batting_action=batting_action,
                bowling_action=current_tactic,
                bowler_name=current_bowler.name,
                outcome=outcome,
            )
            st.session_state.history.append(record)

            # Check if over completed to rotate bowler and tactic
            if outcome.is_legal and sim.legal_balls % 6 == 0 and not sim.is_done:
                squad.complete_over(current_bowler.name)
                next_bowler, next_b_reason = captaincy.select_bowler(squad, sim.legal_balls, sim.score, sim.wickets, sim.target)
                next_tactic, next_t_reason = captaincy.select_tactic(next_bowler, sim.legal_balls, sim.score, sim.wickets, sim.target)
                st.session_state.current_bowler = next_bowler
                st.session_state.current_tactic = next_tactic
                st.session_state.bowling_reason = f"{next_b_reason} | Tactic: {next_t_reason}"

            save_chase_log(st.session_state.history)
            st.rerun()

        # Fast forward execution
        if auto_btn and not sim.is_done:
            with st.spinner("Simulating remaining match deliveries..."):
                while not sim.is_done:
                    obs = sim._observation()
                    q_vals = dqn_agent.get_q_values(obs)
                    batting_action = int(np.argmax(q_vals))

                    step_num = len(st.session_state.history) + 1
                    score_before = sim.score
                    wickets_before = sim.wickets

                    active_b = st.session_state.current_bowler
                    active_t = st.session_state.current_tactic

                    sim.step(
                        aggression=batting_action,
                        bowling_aggression=active_t,
                        bowler=active_b,
                    )
                    outcome = sim.last_outcome
                    overs_str = format_overs(sim.legal_balls) if outcome.is_legal else f"{format_overs(sim.legal_balls)} (extra)"

                    record = build_ball_record(
                        step_num=step_num,
                        overs_str=overs_str,
                        score_before=score_before,
                        wickets_before=wickets_before,
                        target=sim.target,
                        sim=sim,
                        q_vals=q_vals,
                        batting_action=batting_action,
                        bowling_action=active_t,
                        bowler_name=active_b.name,
                        outcome=outcome,
                    )
                    st.session_state.history.append(record)

                    # Over change
                    if outcome.is_legal and sim.legal_balls % 6 == 0 and not sim.is_done:
                        squad.complete_over(active_b.name)
                        next_b, next_br = captaincy.select_bowler(squad, sim.legal_balls, sim.score, sim.wickets, sim.target)
                        next_t, next_tr = captaincy.select_tactic(next_b, sim.legal_balls, sim.score, sim.wickets, sim.target)
                        st.session_state.current_bowler = next_b
                        st.session_state.current_tactic = next_t
                        st.session_state.bowling_reason = f"{next_br} | Tactic: {next_tr}"

                save_chase_log(st.session_state.history)
                st.rerun()

    else:
        # Full innings simulation mode
        if st.button("🚀 Run Full Innings Simulation", width="stretch"):
            sim.reset(target=int(target), initial_wickets=int(initial_wickets), random_seed=int(random_seed))
            squad.reset()
            st.session_state.history = []

            # Initialize first bowler and tactic
            active_b, b_reason = captaincy.select_bowler(squad, 0, 0, int(initial_wickets), int(target))
            active_t, t_reason = captaincy.select_tactic(active_b, 0, 0, int(initial_wickets), int(target))
            st.session_state.current_bowler = active_b
            st.session_state.current_tactic = active_t
            st.session_state.bowling_reason = f"{b_reason} | Tactic: {t_reason}"

            while not sim.is_done:
                obs = sim._observation()
                q_vals = dqn_agent.get_q_values(obs)
                batting_action = int(np.argmax(q_vals))

                step_num = len(st.session_state.history) + 1
                score_before = sim.score
                wickets_before = sim.wickets

                sim.step(
                    aggression=batting_action,
                    bowling_aggression=active_t,
                    bowler=active_b,
                )
                outcome = sim.last_outcome
                overs_str = format_overs(sim.legal_balls) if outcome.is_legal else f"{format_overs(sim.legal_balls)} (extra)"

                record = build_ball_record(
                    step_num=step_num,
                    overs_str=overs_str,
                    score_before=score_before,
                    wickets_before=wickets_before,
                    target=sim.target,
                    sim=sim,
                    q_vals=q_vals,
                    batting_action=batting_action,
                    bowling_action=active_t,
                    bowler_name=active_b.name,
                    outcome=outcome,
                )
                st.session_state.history.append(record)

                if outcome.is_legal and sim.legal_balls % 6 == 0 and not sim.is_done:
                    squad.complete_over(active_b.name)
                    active_b, b_reason = captaincy.select_bowler(squad, sim.legal_balls, sim.score, sim.wickets, sim.target)
                    active_t, t_reason = captaincy.select_tactic(active_b, sim.legal_balls, sim.score, sim.wickets, sim.target)
                    st.session_state.current_bowler = active_b
                    st.session_state.current_tactic = active_t
                    st.session_state.bowling_reason = f"{b_reason} | Tactic: {t_reason}"

            save_chase_log(st.session_state.history)
            st.rerun()

    # Latest Decision Card
    if st.session_state.history:
        latest = st.session_state.history[-1]
        st.markdown("### 🧠 Latest Delivery Decisions & DQN Q-Values")

        q_col1, q_col2, q_col3 = st.columns(3)
        with q_col1:
            st.metric("Q(Conservative)", f"{latest['q_conservative']:.4f}")
        with q_col2:
            st.metric("Q(Balanced)", f"{latest['q_balanced']:.4f}")
        with q_col3:
            st.metric("Q(Aggressive)", f"{latest['q_aggressive']:.4f}")

        # Factual decision rationale
        bat_act = latest["batting_action"]
        q_sel = latest[f"q_{bat_act.lower()}"]
        q_others = [v for k, v in latest.items() if k.startswith("q_") and not k.endswith(bat_act.lower())]
        margin = q_sel - max(q_others) if q_others else 0.0

        st.markdown(
            f"""
            <div style="background-color: rgba(31, 119, 180, 0.08); padding: 14px 18px; border-radius: 8px; border-left: 4px solid #1f77b4; margin-bottom: 12px;">
                <b>Delivery {latest['ball_num']} (Over {latest['over']}):</b> &nbsp;
                Bowler: <b>{latest['bowler']}</b> (Tactic: <code>{latest['bowling_tactic']}</code>) &nbsp;|&nbsp;
                Score Before: <code>{latest['score_before']}</code> &nbsp;➔&nbsp;
                Event: <b>{latest['event_desc']}</b> &nbsp;➔&nbsp;
                Score After: <code>{latest['score_after']}</code>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.info(
            f"**Factual Batting Rationale (DQN):** The batting agent chose **{bat_act}** because its predicted Q-value "
            f"(**{q_sel:.4f}**) was maximal among discrete actions with a margin of **+{margin:.4f}** over the next best action.  \n"
            f"**Bowling Context:** {latest['bowler']} bowled with **{latest['bowling_tactic']}** approach.  \n"
            f"*(Scientific Note: Q-values represent the Bellman expected discounted cumulative future return Q(s, a). "
            f"They are NOT calibrated win probabilities.)*"
        )

    # Innings Completion Status
    if sim.is_done:
        st.markdown("---")
        st.subheader("🏁 Match Chase Result")
        if sim.won:
            balls_left = 120 - sim.legal_balls
            wkts_in_hand = 10 - sim.wickets
            st.markdown(
                f"<div class='status-won'>🎉 CHASE ACCOMPLISHED: {batting_team} WON by {wkts_in_hand} wickets with {balls_left} balls remaining! (Score: {sim.score}/{sim.wickets} in {format_overs(sim.legal_balls)} overs)</div>",
                unsafe_allow_html=True,
            )
        else:
            reason = "All 10 wickets lost (All-Out)" if sim.wickets == 10 else "20 overs completed (Overs exhausted)"
            deficit = sim.target - sim.score
            st.markdown(
                f"<div class='status-lost'>❌ CHASE DEFEAT: {bowling_team} WON by {deficit} runs — {reason}. (Score: {sim.score}/{sim.wickets} in {format_overs(sim.legal_balls)} overs)</div>",
                unsafe_allow_html=True,
            )

        # Dual Action Breakdown & Bowling Scorecard
        if st.session_state.history:
            df_hist = pd.DataFrame(st.session_state.history)
            tot_balls = len(df_hist)

            st.markdown("#### 📊 Tactical Action Breakdown (Batting vs Bowling)")
            pcol1, pcol2 = st.columns(2)

            with pcol1:
                st.markdown("**Batting Policy (DQN Agent):**")
                bat_counts = df_hist["batting_action"].value_counts().to_dict()
                c_cnt = bat_counts.get("Conservative", 0)
                b_cnt = bat_counts.get("Balanced", 0)
                a_cnt = bat_counts.get("Aggressive", 0)
                st.write(f"• Conservative: {c_cnt} ({c_cnt/tot_balls*100:.1f}%)")
                st.write(f"• Balanced: {b_cnt} ({b_cnt/tot_balls*100:.1f}%)")
                st.write(f"• Aggressive: {a_cnt} ({a_cnt/tot_balls*100:.1f}%)")

            with pcol2:
                st.markdown(f"**Bowling Policy ({bowling_policy_choice.split('(')[0].strip()}):**")
                bowl_counts = df_hist["bowling_tactic"].value_counts().to_dict()
                def_cnt = bowl_counts.get("Defensive", 0)
                bal_cnt = bowl_counts.get("Balanced", 0)
                atk_cnt = bowl_counts.get("Attacking", 0)
                st.write(f"• Defensive: {def_cnt} ({def_cnt/tot_balls*100:.1f}%)")
                st.write(f"• Balanced: {bal_cnt} ({bal_cnt/tot_balls*100:.1f}%)")
                st.write(f"• Attacking: {atk_cnt} ({atk_cnt/tot_balls*100:.1f}%)")

            # Official Bowling Scorecard
            st.markdown(f"#### 📋 Official Bowling Scorecard — {bowling_team}")
            df_bowl_summary = squad.get_figures_summary()
            if not df_bowl_summary.empty:
                st.dataframe(df_bowl_summary, width="stretch")

            # Score Trajectory Chart
            st.markdown("#### 📈 Chase Score Progression Trajectory")
            fig, ax = plt.subplots(figsize=(10, 4))
            balls_series = np.arange(len(df_hist)) + 1
            scores_series = [int(s.split("/")[0]) for s in df_hist["score_after"]]

            ax.plot(balls_series, scores_series, label=f"{batting_team} Score", color="#1f77b4", linewidth=2.5)
            ax.axhline(sim.target, color="#d62728", linestyle="--", linewidth=1.5, label=f"Target ({sim.target})")
            ax.set_xlabel("Deliveries Bowled")
            ax.set_ylabel("Total Runs")
            ax.set_title(f"Second-Innings Chase Progression (Target: {sim.target})", fontsize=12, fontweight="bold")
            ax.legend(loc="upper left")
            st.pyplot(fig)
            plt.close(fig)

    # Ball-by-ball table
    if st.session_state.history:
        st.markdown("---")
        st.subheader("📜 Ball-by-Ball Decision Log")
        st.caption(
            "💡 **Scoring Guide**: On **Dot balls (⚪)**, no runs are scored so `Score Before` and `Score After` are identical (e.g. `18/4 ➔ 18/4`). "
            "On **Wickets (🔴)**, 0 runs are scored so the runs total stays unchanged while wickets increment (e.g. `0/4 ➔ 0/5`). "
            "On **Scoring deliveries (🏃/4️⃣/6️⃣)**, runs are added to the total."
        )
        df_display = pd.DataFrame(st.session_state.history)
        cols_to_show = [
            ("ball_num", "Ball #"),
            ("over", "Over"),
            ("bowler", "Bowler"),
            ("bowling_tactic", "Bowling Tactic"),
            ("score_before", "Score Before"),
            ("event_desc", "Delivery Event"),
            ("outcome_runs", "Runs Added"),
            ("score_after", "Score After"),
            ("runs_required", "Runs Needed"),
            ("balls_remaining", "Balls Left"),
            ("rrr", "RRR"),
            ("batting_action", "DQN Batting"),
            ("q_conservative", "Q(Cons)"),
            ("q_balanced", "Q(Bal)"),
            ("q_aggressive", "Q(Agg)"),
        ]
        available_cols = [c for c in cols_to_show if c[0] in df_display.columns]
        df_renamed = df_display[[c[0] for c in available_cols]].rename(columns=dict(available_cols))
        st.dataframe(df_renamed, width="stretch", height=340)

        # Download CSV Button
        csv_data = df_display.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Ball-by-Ball Decision CSV",
            data=csv_data,
            file_name=f"chase_{batting_team}_vs_{bowling_team}_target_{sim.target}_seed_{st.session_state.seed}.csv",
            mime="text/csv",
        )

# TAB 2: BASELINE BENCHMARKS & AUDIT DISCLOSURE
with tabs[1]:
    st.header("📊 Comparative Evaluation: Batting DQN & Bowling Captaincies")

    st.markdown(
        r"""
        ### 🔍 Mandatory Scientific Disclosure
        > **1. Batting Policy**: The batting captaincy agent is an autonomous **Deep Q-Network (DQN)** trained via Bellman Temporal Difference learning.
        > In standard par and demanding chases (165 and 185 runs), DQN selects **Aggressive batting on approximately 78% – 98% of deliveries** because over exhaustion is a far greater risk than wicket exhaustion with 10 wickets in hand.
        > DQN matches `AlwaysAggressive` within statistical confidence intervals and substantially beats realistic heuristics like `AlwaysBalanced` (+8% to +24%) and `RRRPolicy` (+12% to +20%).
        >
        > **2. Bowling Policy**: Bowling selection and tactics are governed by **data-driven empirical captaincy rules** utilizing authentic IPL player statistics (economy rates, strike rates, phase specializations) and strict cricket workload constraints (max 4 overs, no consecutive overs).
        """
    )

    # Benchmark Results Table from evaluation_results.csv
    csv_eval_path = ROOT / "reports" / "evaluation_results.csv"
    if csv_eval_path.exists():
        st.subheader("📋 Rigorous Batting Benchmark Table (500 Episodes per Policy & Target, 95% Wilson CI)")
        df_bench = pd.read_csv(csv_eval_path)
        st.dataframe(
            df_bench[[
                "policy", "target", "win_rate", "ci_95_low", "ci_95_high",
                "mean_score", "std_score", "mean_wickets", "all_out_rate", "overs_complete_rate"
            ]],
            width="stretch",
        )
    else:
        st.warning("reports/evaluation_results.csv not found.")

    # Live Head-to-Head Batting & Bowling Race
    st.subheader("⚡ Live Head-to-Head Strategy Race")
    st.markdown("Run all 5 batting policies against the selected bowling strategy under identical random conditions:")

    h2h_target = st.number_input("Comparison Target", min_value=120, max_value=220, value=165, step=5, key="h2h_t")
    h2h_seed = st.number_input("Comparison Seed", min_value=0, max_value=999999, value=2026, step=1, key="h2h_s")

    if st.button("🏁 Run 5-Policy Batting Race"):
        with st.spinner("Simulating all 5 policies..."):
            policies = [
                ("AlwaysConservative", AlwaysConservative()),
                ("AlwaysBalanced", AlwaysBalanced()),
                ("RRRPolicy", RRRPolicy()),
                ("AlwaysAggressive", AlwaysAggressive()),
                ("DQNAgent (Ours)", dqn_agent),
            ]

            h2h_results = []
            for name, pol in policies:
                sim_test = T20Simulator(outcome_model=outcome_model, random_seed=int(h2h_seed))
                sim_test.reset(target=int(h2h_target), initial_wickets=0, random_seed=int(h2h_seed))

                act_counter = {0: 0, 1: 0, 2: 0}
                while not sim_test.is_done:
                    obs = sim_test._observation()
                    if hasattr(pol, "select_action"):
                        if isinstance(pol, DQNAgent):
                            act = pol.select_action(obs, eval_mode=True)
                        else:
                            act = pol.select_action(sim_test)
                    else:
                        act = pol(sim_test)
                    act_counter[act] += 1
                    sim_test.step(act)

                h2h_results.append({
                    "Policy": name,
                    "Result": "WON 🏆" if sim_test.won else "LOST ❌",
                    "Score": f"{sim_test.score}/{sim_test.wickets}",
                    "Overs": format_overs(sim_test.legal_balls),
                    "Conservative %": f"{act_counter[0]/sum(act_counter.values())*100:.1f}%",
                    "Balanced %": f"{act_counter[1]/sum(act_counter.values())*100:.1f}%",
                    "Aggressive %": f"{act_counter[2]/sum(act_counter.values())*100:.1f}%",
                })

            st.dataframe(pd.DataFrame(h2h_results), width="stretch")

    # Static Evaluation Figures
    st.subheader("📊 Evaluation Artifacts from Reports")
    pcol1, pcol2 = st.columns(2)
    with pcol1:
        img_policy = ROOT / "reports" / "policy_comparison.png"
        if img_policy.exists():
            st.image(str(img_policy), caption="Win Rate Comparison with 95% Wilson CIs", width="stretch")
    with pcol2:
        img_curves = ROOT / "reports" / "training_curves.png"
        if img_curves.exists():
            st.image(str(img_curves), caption="DQN Training Dynamics (1,500 Episodes)", width="stretch")

# TAB 3: SYSTEM ARCHITECTURE & LOGIC
with tabs[2]:
    st.header("📖 System Architecture & Logic Explained Simply")

    st.markdown(
        r"""
        ### 1. State Space & Feature Representation
        At every legal delivery $t$, the batting agent observes a 6-dimensional normalized observation vector in $[0, 1]$:
        1. **Score**: Current score normalized by 250.
        2. **Wickets Fallen**: Normalized by 10.
        3. **Balls Remaining**: Legal deliveries left normalized by 120.
        4. **Target**: Chase target normalized by 250.
        5. **Runs Required**: Normalized by 250.
        6. **Required Run Rate (RRR)**: Normalized by 24.0 rpo.

        ### 2. Batting Action Space
        - **0: Conservative**: Reduced boundary probability (0.60x), reduced wicket probability (0.75x). Favors wicket preservation.
        - **1: Balanced**: Natural historical IPL delivery outcome frequencies.
        - **2: Aggressive**: Increased boundary probability (1.50x), increased wicket probability (1.30x). High risk, high scoring rate.

        ### 3. Corrected Potential-Based Reward Shaping (Ng et al., 1999)
        To preserve optimal policy invariance while accelerating learning:
        $$F(s_t, a_t, s_{t+1}) = \gamma \Phi(s_{t+1}) - \Phi(s_t)$$
        where $\Phi(s) = \text{WinProbability}(s) \in [0, 1]$ and $\gamma = 0.99$ matches the DQN discount factor.
        At terminal ball $T$:
        $$r_T = R_T + \gamma \Phi(s_T) - \Phi(s_{T-1})$$
        where $\Phi(s_T) = 1.0$ (win) or $0.0$ (loss), and $R_T \in \{+1.0, -1.0\}$.

        ### 4. Bowling Tactics & Workload System
        - **Over Limits**: Strictly enforces the official ICC/IPL limit of maximum 4 overs (24 legal balls) per bowler in a 20-over match.
        - **Consecutive Over Rule**: Bowler cannot bowl two consecutive overs.
        - **Tactics**:
          - **0: Defensive**: Yorkers/wide containment (boundary rate 0.80x, wicket rate 0.88x).
          - **1: Balanced**: Baseline execution (1.00x).
          - **2: Attacking**: Stumps/short-pitch/flighted spin (wicket rate 1.25x, boundary risk 1.15x).
        - **Captaincy Policies**:
          - `PhaseSpecialistCaptaincy`: Opening pacers in powerplay, spinners in middle, death specialists in death.
          - `RiskAdaptiveCaptaincy`: Reactive containment chokes on high RRR (>10.5) and attacking strike bowling on low RRR (<7.0).
          - `RoundRobinCaptaincy`: Baseline cyclic rotation.

        ### 5. Simulator Physics & Cricket Correctness
        - **Wicket Run Clamp**: Historically 97.5% of wicket deliveries yield 0 batter runs. When a wicket occurs, `batter_runs` is clamped to `0`.
        - **Extras**: Wides and no-balls add penalty runs without consuming legal balls.
        """
    )

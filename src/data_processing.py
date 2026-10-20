import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "ipl_json"
OUTPUT_DIR = ROOT / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def is_legal_delivery(extras):
    """Wides and no-balls do not count as legal balls."""
    return not ("wides" in extras or "noballs" in extras)


def is_counted_wicket(wicket):
    """Retired hurt is not counted as a wicket."""
    return wicket.get("kind", "").lower() != "retired hurt"


def parse_match(path):
    with path.open("r", encoding="utf-8") as f:
        match = json.load(f)

    info = match.get("info", {})
    innings_list = match.get("innings", [])
    match_id = path.stem

    teams = info.get("teams", [])
    outcome = info.get("outcome", {})
    winner = outcome.get("winner")

    match_rows = []
    delivery_rows = []
    batting_stats = {}
    bowling_stats = {}
    team_stats = {}

    innings_totals = {}

    # First pass: calculate each innings' total score.
    for innings_no, innings in enumerate(innings_list, start=1):
        total = 0

        for over_data in innings.get("overs", []):
            for delivery in over_data.get("deliveries", []):
                total += delivery.get("runs", {}).get("total", 0)

        innings_totals[innings_no] = total

    # Second pass: build the state before and after every delivery.
    for innings_no, innings in enumerate(innings_list, start=1):
        batting_team = innings.get("team")
        bowling_team = next( (team for team in teams if team != batting_team), None,)

        target = (
            innings_totals.get(1, 0) + 1
            if innings_no == 2
            else None
        )

        score = 0
        wickets_lost = 0
        legal_balls = 0

        for over_data in innings.get("overs", []):
            over_number = over_data.get("over", 0)

            for ball_number, delivery in enumerate(
                over_data.get("deliveries", []), start=1
            ):
                batter = delivery.get("batter")
                bowler = delivery.get("bowler")
                non_striker = delivery.get("non_striker")

                runs = delivery.get("runs", {})
                extras = delivery.get("extras", {})
                wickets = delivery.get("wickets", [])

                batter_runs = runs.get("batter", 0)
                extra_runs = runs.get("extras", 0)
                total_runs = runs.get("total", 0)

                legal = is_legal_delivery(extras)
                wickets_this_ball = sum(
                    is_counted_wicket(w) for w in wickets
                )

                # Snapshot before this delivery.
                score_before = score
                wickets_before = wickets_lost
                balls_before = legal_balls

                delivery_rows.append({
                    "match_id": match_id,
                    "innings": innings_no,
                    "over": over_number,
                    "delivery_in_over": ball_number,
                    "batting_team": batting_team,
                    "bowling_team": bowling_team,
                    "batter": batter,
                    "bowler": bowler,
                    "non_striker": non_striker,
                    "target": target,
                    "runs_before": score_before,
                    "wickets_before": wickets_before,
                    "legal_balls_before": balls_before,
                    "runs_required_before": (
                        max(target - score_before, 0)
                        if target is not None else None
                    ),
                    "batter_runs": batter_runs,
                    "extra_runs": extra_runs,
                    "total_runs": total_runs,
                    "extras_type": ",".join(extras.keys()),
                    "legal_delivery": legal,
                    "wickets_on_delivery": wickets_this_ball,
                    "runs_after": score_before + total_runs,
                    "wickets_after": (
                        wickets_before + wickets_this_ball
                    ),
                    "legal_balls_after": (
                        balls_before + int(legal)
                    ),
                })

                score += total_runs
                wickets_lost += wickets_this_ball
                legal_balls += int(legal)

                # Batting statistics.
                bat = batting_stats.setdefault(
                    batter,
                    {
                        "player": batter,
                        "runs": 0,
                        "balls_faced": 0,
                        "fours": 0,
                        "sixes": 0,
                    },
                )

                bat["runs"] += batter_runs

                # A wide does not count as a ball faced.
                if "wides" not in extras:
                    bat["balls_faced"] += 1

                if batter_runs == 4:
                    bat["fours"] += 1
                elif batter_runs == 6:
                    bat["sixes"] += 1

                # Bowling statistics.
                bowl = bowling_stats.setdefault(
                    bowler,
                    {
                        "player": bowler,
                        "runs_conceded": 0,
                        "legal_balls": 0,
                        "wickets": 0,
                    },
                )

                # Byes and leg-byes are not charged to the bowler.
                uncharged = (
                    extras.get("byes", 0)
                    + extras.get("legbyes", 0)
                    + extras.get("penalty", 0)
                )

                bowl["runs_conceded"] += total_runs - uncharged
                bowl["legal_balls"] += int(legal)

                # Basic estimate of wickets credited to the bowler.
                not_bowler_wickets = {
                    "run out",
                    "retired hurt",
                    "retired out",
                    "obstructing the field",
                }

                bowl["wickets"] += sum(
                    w.get("kind", "").lower()
                    not in not_bowler_wickets
                    for w in wickets
                )

                # Team totals by innings.
                team_stats.setdefault(
                    batting_team,
                    {
                        "team": batting_team,
                        "innings": 0,
                        "runs": 0,
                        "wickets": 0,
                    },
                )

        team_stats[batting_team]["innings"] += 1
        team_stats[batting_team]["runs"] += score
        team_stats[batting_team]["wickets"] += wickets_lost

        match_rows.append({
            "match_id": match_id,
            "date": ",".join(map(str, info.get("dates", []))),
            "season": info.get("season"),
            "venue": info.get("venue"),
            "innings": innings_no,
            "batting_team": batting_team,
            "bowling_team": bowling_team,
            "runs": score,
            "wickets": wickets_lost,
            "legal_balls": legal_balls,
            "winner": winner,
            "match_type": info.get("match_type"),
        })

    return delivery_rows, match_rows, batting_stats, bowling_stats, team_stats


def main():
    files = sorted(RAW_DIR.rglob("*.json"))

    if not files:
        raise FileNotFoundError(
            f"No JSON files found in {RAW_DIR}. "
            "Download and extract the Cricsheet IPL dataset first."
        )

    all_deliveries = []
    all_innings = []
    batting_stats = {}
    bowling_stats = {}
    team_stats = {}

    errors = []

    for path in files:
        try:
            deliveries, innings, bat_stats, bowl_stats, teams = (parse_match(path))

            all_deliveries.extend(deliveries)
            all_innings.extend(innings)

            for player, stats in bat_stats.items():
                if player not in batting_stats:
                    batting_stats[player] = stats.copy()
                else:
                    dest = batting_stats[player]
                    for key in ("runs", "balls_faced", "fours", "sixes"):
                        dest[key] += stats[key]

            for player, stats in bowl_stats.items():
                if player not in bowling_stats:
                    bowling_stats[player] = stats.copy()
                else:
                    dest = bowling_stats[player]
                    for key in (
                        "runs_conceded", "legal_balls", "wickets"
                    ):
                        dest[key] += stats[key]

            for team, stats in teams.items():
                if team not in team_stats:
                    team_stats[team] = stats.copy()
                else:
                    dest = team_stats[team]
                    for key in ("innings", "runs", "wickets"):
                        dest[key] += stats[key]

        except (ValueError, KeyError, TypeError) as exc:
            errors.append((path.name, str(exc)))

    pd.DataFrame(all_deliveries).to_csv(OUTPUT_DIR/"deliveries.csv", index=False)
    pd.DataFrame(all_innings).to_csv(OUTPUT_DIR / "match_summary.csv", index=False)
    pd.DataFrame(batting_stats.values()).to_csv(OUTPUT_DIR /"player_batting_stats.csv", index=False)
    pd.DataFrame(bowling_stats.values()).to_csv(OUTPUT_DIR / "player_bowling_stats.csv", index=False)
    pd.DataFrame(team_stats.values()).to_csv(OUTPUT_DIR/ "team_stats.csv", index=False)

    print(f"JSON files discovered: {len(files)}")
    print(f"Delivery rows: {len(all_deliveries)}")
    print(f"Innings rows: {len(all_innings)}")
    print(f"Teams: {len(team_stats)}")
    print(f"Output directory: {OUTPUT_DIR}")

    if errors:
        print(f"Files with parsing errors: {len(errors)}")
        for name, error in errors[:10]:
            print(f"  {name}: {error}")


if __name__ == "__main__":
    main()
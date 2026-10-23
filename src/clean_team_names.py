from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"

# Map historical names to a canonical franchise name.
TEAM_MAPPING = {
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Rising Pune Supergiant": "Rising Pune Supergiants",
}

def canonicalize(series):
    return series.replace(TEAM_MAPPING)

def main():
    # Clean delivery records without modifying the original CSV.
    deliveries_path = DATA_DIR/"deliveries.csv"
    deliveries = pd.read_csv(deliveries_path)

    for column in ["batting_team", "bowling_team"]:
        deliveries[f"{column}_canonical"] = canonicalize(
            deliveries[column]
        )

    deliveries.to_csv(DATA_DIR/"deliveries_clean.csv", index=False)

    # Clean innings summaries.
    summary_path = DATA_DIR/ "match_summary.csv"
    summary = pd.read_csv(summary_path)

    for column in ["batting_team", "bowling_team", "winner"]:
        if column in summary.columns:
            summary[f"{column}_canonical"] = canonicalize(
                summary[column]
            )

    summary.to_csv(DATA_DIR / "match_summary_clean.csv", index=False)

    # Aggregate team statistics after standardizing names.
    team_path = DATA_DIR / "team_stats.csv"
    teams = pd.read_csv(team_path)
    teams["team_canonical"] = canonicalize(teams["team"])

    canonical_stats = ( teams.groupby("team_canonical", as_index=False).agg({
            "innings": "sum",
            "runs": "sum",
            "wickets": "sum",
        }).rename(columns={"team_canonical": "team"}).sort_values("team")
    )

    canonical_stats.to_csv(DATA_DIR / "team_stats_canonical.csv", index=False)

    print("Team-name cleaning complete.")
    print("Original team names:", teams["team"].nunique())
    print("Canonical team names:", canonical_stats["team"].nunique())
    print("\nCanonical team statistics:")
    print(canonical_stats.to_string(index=False))

    print("\nCreated:")
    print(" - data/processed/deliveries_clean.csv")
    print(" - data/processed/match_summary_clean.csv")
    print(" - data/processed/team_stats_canonical.csv")

if __name__ == "__main__":
    main()
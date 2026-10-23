# calculate each bowler's legal balls per innings and flag overs that exceed the standard IPL limit

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"

# Standard IPL innings: maximum 4 overs per bowler.
MAX_OVERS = 4
BALLS_PER_OVER = 6
MAX_LEGAL_BALLS = MAX_OVERS * BALLS_PER_OVER


def main():
    path = DATA_DIR / "deliveries_clean.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run src/clean_team_names.py first."
        )

    df = pd.read_csv(path)

    required = {"match_id", "innings", "bowler", "legal_delivery", "bowling_team_canonical",}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Convert legal_delivery robustly if pandas read it as text.
    df["legal_delivery"] = (
        df["legal_delivery"].astype(str).str.lower().map({"true": True, "false": False, "1": True, "0": False})
    )
    if df["legal_delivery"].isna().any():
        raise ValueError("Unexpected values in legal_delivery.")

    # Each bowler's workload within a match innings.
    workload = (
        df.groupby(["match_id", "innings", "bowling_team_canonical", "bowler"], dropna=False,).agg(
            legal_balls=("legal_delivery", "sum"),
            deliveries=("legal_delivery", "size"),
        ).reset_index()
    )

    workload["legal_balls"] = workload["legal_balls"].astype(int)
    workload["completed_overs"] = workload["legal_balls"] // BALLS_PER_OVER
    workload["balls_in_partial_over"] = (
        workload["legal_balls"] % BALLS_PER_OVER
    )
    workload["exceeds_standard_4_over_limit"] = (
        workload["legal_balls"] > MAX_LEGAL_BALLS
    )

    output = DATA_DIR / "bowler_workload.csv"
    workload.to_csv(output, index=False)

    print("Bowler workload rows:", len(workload))
    print("Rows exceeding 24 legal balls:",
          workload["exceeds_standard_4_over_limit"].sum())
    print("\nLargest workloads:")
    print(
        workload.sort_values("legal_balls", ascending=False)
        .head(10)
        .to_string(index=False)
    )
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
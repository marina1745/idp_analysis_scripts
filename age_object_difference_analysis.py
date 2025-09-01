"""
Task: Analyze how participants' age relates to accuracy in Task 2 Part 2.
For each participant:
- Extract age from final_result*.json
- Compare reference object positions (Specific-Task2.csv) with final user
  positions (EyeTracking-Task2 Part 2.csv)
- Compute the average Euclidean distance (object placement error)

Rules / Data Quality:
- A participant is SKIPPED if:
  * Required files are missing
  * First 50 frames of Task 2 Part 2 have GazeStatus == "INVALID"
  * Positions cannot be parsed or no valid object positions exist
- We keep track of how many were processed vs skipped.

Outputs:
- task2_results.csv with participant, age, avg_object_error
- age_vs_error.png regression plot (Age vs Object Error)
"""

import os
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from pathlib import Path
from typing import Optional

# -----------------------------------
# Configuration
# -----------------------------------
BASE_DIR = Path(r"D:/Uni/WS2425/IDP/logs_filtered/logs")
DROP_LAST_FRAMES = 30
GAZE_INVALID_CHECK_N = 50

# -----------------------------------
# Helpers
# -----------------------------------
def load_csv(csv_path: Path) -> pd.DataFrame:
    """Load CSV robustly, clean column names."""
    df = pd.read_csv(csv_path, sep=";", dtype=str, encoding="utf-8", on_bad_lines="skip")
    df.columns = [c.strip().replace("\ufeff", "") for c in df.columns]
    return df

def first_n_invalid(df: pd.DataFrame, n: int) -> bool:
    """Return True if first n frames have GazeStatus == 'INVALID'."""
    if "GazeStatus" not in df.columns:
        return False
    first = df["GazeStatus"].head(n).astype(str).str.upper()
    return len(first) > 0 and all(s == "INVALID" for s in first)

def parse_position(val: str) -> Optional[np.ndarray]:
    """Convert '(x,y,z)' string into numpy array, or None if invalid."""
    try:
        return np.array(eval(val))
    except Exception:
        return None

# -----------------------------------
# Main
# -----------------------------------
data = []
processed = 0
skipped = 0

for folder in sorted(BASE_DIR.iterdir()):
    if not folder.is_dir():
        continue
    participant = folder.name

    # Find files
    json_path = next((p for p in folder.glob("final_result*.json")), None)
    reference_path = next((p for p in folder.glob("*Specific-Task2.csv")), None)
    task2_part2_path = next((p for p in folder.glob("*EyeTracking-Task2 Part 2.csv")), None)

    if not json_path or not reference_path or not task2_part2_path:
        print(f"[{participant}] Skipped: missing file(s)")
        skipped += 1
        continue

    # Extract age
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            j = json.load(f)
        age = int(j["data_object5"]["answers"][0])
    except Exception as e:
        print(f"[{participant}] Skipped: failed to read age ({e})")
        skipped += 1
        continue

    # Load CSVs
    try:
        ref_df = load_csv(reference_path)
        df2 = load_csv(task2_part2_path)
    except Exception as e:
        print(f"[{participant}] Skipped: CSV load error ({e})")
        skipped += 1
        continue

    # GazeStatus invalid check
    if first_n_invalid(df2, GAZE_INVALID_CHECK_N):
        print(f"[{participant}] Skipped: first 10 frames INVALID in Task 2 Part 2")
        skipped += 1
        continue

    # Remove last frames
    if len(df2) > DROP_LAST_FRAMES:
        df2 = df2.iloc[:-DROP_LAST_FRAMES]

    object_errors = []
    for col in ref_df.columns:
        if not col.endswith("_Position"):
            continue
        if col not in df2.columns:
            print(f"[{participant}] Warning: column {col} missing")
            continue

        # Reference position
        ref_val = ref_df[col].dropna().iloc[0] if not ref_df[col].dropna().empty else None
        ref_pos = parse_position(ref_val) if ref_val else None
        if ref_pos is None:
            continue

        # User final positions
        user_vals = df2[col].dropna().apply(parse_position).dropna()
        if user_vals.empty:
            continue
        user_pos = np.mean(user_vals.tolist(), axis=0)

        # Distance
        dist = np.linalg.norm(ref_pos - user_pos)
        object_errors.append(dist)

    if not object_errors:
        print(f"[{participant}] Skipped: no valid object positions")
        skipped += 1
        continue

    avg_error = float(np.mean(object_errors))
    print(f"[{participant}] ✔ Processed (age={age}, avg_error={avg_error:.3f})")
    data.append({
        "participant": participant,
        "age": age,
        "avg_object_error": avg_error
    })
    processed += 1

# Save results
results_df = pd.DataFrame(data)
results_df.to_csv("task2_results.csv", index=False)
print("Saved to task2_results.csv")

# Regression plot
if not results_df.empty:
    df = results_df.dropna()
    X = df[["age"]]
    y = df["avg_object_error"]

    model = LinearRegression()
    model.fit(X, y)

    plt.figure(figsize=(8, 5))
    sns.regplot(x="age", y="avg_object_error", data=df, ci=None, scatter_kws={"alpha":0.7})
    plt.title("Linear Regression: Age vs Object Error (Task 2)")
    plt.xlabel("Age")
    plt.ylabel("Average Object Error (Euclidean distance)")
    plt.tight_layout()
    plt.savefig("age_vs_error.png")
    plt.close()
    print("Saved plot to age_vs_error.png")
else:
    print("No valid data for regression plot")

print("—— Summary ——")
print(f"Participants processed: {processed}")
print(f"Participants skipped:   {skipped}")

"""
Task: Extract each participant's age (from final_result*.json) and the time spent
in Task 2 and Task 2 Part 2 (from the respective CSV logs), then analyze the
relationship between age and task duration via simple linear regression.

Rules / Data Quality:
- A participant is SKIPPED if, for Task 2 or Task 2 Part 2, the first 10 frames
  have GazeStatus == "INVALID". (If either file violates this, we skip them.)
- We also skip participants missing required files/columns or with too few rows
  to compute a stable duration (we use CaptureTime at row 30 and -30).
- We count and print how many participants were processed vs. skipped.

Outputs:
- regression_task2_age.png — plot of linear fits for Task 2 and Task 2 Part 2
- task2_age_analysis.csv  — cleaned table of participant, age, and durations
- Console prints with processed / skipped counts and reasons per participant
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
from typing import Optional
from pathlib import Path

# -----------------------------------
# Configuration
# -----------------------------------
BASE_DIR = Path(r"D:/Uni/WS2425/IDP/logs_filtered/logs")
DROP_FRAMES_EACH_END = 30      # we read time at +30 and -30
MIN_ROWS_NEEDED = DROP_FRAMES_EACH_END * 2 + 1  # ensure we can index both ends
GAZE_INVALID_CHECK_N = 50      # "first N frames INVALID" -> skip participant

# -----------------------------------
# Helpers
# -----------------------------------

def load_csv(csv_path: Path) -> pd.DataFrame:
    """Load CSV with consistent cleanup; return DataFrame (may raise)."""
    df = pd.read_csv(
        csv_path,
        sep=";",
        dtype=str,
        encoding="utf-8",
        on_bad_lines="skip"
    )
    # strip BOM and whitespace from column names
    df.columns = [c.strip().replace("\ufeff", "") for c in df.columns]
    return df

def first_n_invalid(df: pd.DataFrame, n: int) -> bool:
    """Return True if the first n frames have GazeStatus == 'INVALID' (case-insensitive).
       If column missing, we treat as not-invalid-only check (i.e., return False and let other checks handle)."""
    if "GazeStatus" not in df.columns:
        return False
    first = df["GazeStatus"].head(n).astype(str).str.upper()
    # If there are fewer than n rows, this still returns True only if *all available* are INVALID
    return len(first) > 0 and all(s == "INVALID" for s in first)

def extract_duration(csv_path: Path) -> Optional[float]:
    """
    Compute duration (seconds) from a Task 2 CSV:
    - Uses 'CaptureTime' in nanoseconds
    - Reads time at row index +30 and -30 to avoid noisy edges
    Returns float seconds or None if not possible.
    """
    try:
        df = load_csv(csv_path)
        if "CaptureTime" not in df.columns:
            print(f"  ❌ Missing 'CaptureTime' in {csv_path.name}")
            return None

        # Need enough rows to safely index +30 and -30
        if len(df) < MIN_ROWS_NEEDED:
            print(f"  ❌ Too few rows ({len(df)}) in {csv_path.name}; need ≥ {MIN_ROWS_NEEDED}")
            return None

        # Clean and convert to integer nanoseconds
        ct = (
            df["CaptureTime"]
            .str.replace("'", "", regex=False)
            .str.strip()
            .astype(np.int64)
        )

        time_start = ct.iloc[DROP_FRAMES_EACH_END]
        time_end   = ct.iloc[-DROP_FRAMES_EACH_END - 1]
        duration_s = (time_end - time_start) / 1e9
        return float(duration_s)
    except Exception as e:
        print(f"  ❌ Error reading {csv_path.name}: {e}")
        return None

# -----------------------------------
# Main extraction
# -----------------------------------

data = []
processed = 0
skipped = 0

for folder in sorted(BASE_DIR.iterdir()):
    if not folder.is_dir():
        continue

    participant = folder.name

    # Find JSON
    json_file = next((p for p in folder.glob("final_result*.json")), None)
    if not json_file:
        print(f"[{participant}] Skipped: no final_result*.json")
        skipped += 1
        continue

    # Read age
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            j = json.load(f)
        age = int(j["data_object5"]["answers"][0])
    except Exception as e:
        print(f"[{participant}] Skipped: failed to read age from JSON ({e})")
        skipped += 1
        continue

    # Locate Task 2 csvs
    task2_path = next((p for p in folder.glob("*EyeTracking-Task2.csv") if "Part" not in p.name), None)
    task2_part2_path = next((p for p in folder.glob("*EyeTracking-Task2 Part 2.csv")), None)

    if not task2_path or not task2_part2_path:
        print(f"[{participant}] Skipped: missing Task 2 CSV(s) "
              f"(Task2={'OK' if task2_path else 'MISSING'}, Part2={'OK' if task2_part2_path else 'MISSING'})")
        skipped += 1
        continue

    # GazeStatus INVALID check (first 50 frames) – skip if either file fails
    try:
        df_t2 = load_csv(task2_path)
        df_t2p2 = load_csv(task2_part2_path)
    except Exception as e:
        print(f"[{participant}] Skipped: CSV load error ({e})")
        skipped += 1
        continue

    invalid_t2 = first_n_invalid(df_t2, GAZE_INVALID_CHECK_N)
    invalid_t2p2 = first_n_invalid(df_t2p2, GAZE_INVALID_CHECK_N)
    if invalid_t2 or invalid_t2p2:
        reason = []
        if invalid_t2:
            reason.append("Task 2 first 50 frames INVALID")
        if invalid_t2p2:
            reason.append("Task 2 Part 2 first 50 frames INVALID")
        print(f"[{participant}] Skipped: " + " & ".join(reason))
        skipped += 1
        continue

    # Compute durations
    task2_time = extract_duration(task2_path)
    task2_part2_time = extract_duration(task2_part2_path)

    if task2_time is None or task2_part2_time is None:
        print(f"[{participant}] Skipped: could not compute both durations "
              f"(Task2={task2_time}, Part2={task2_part2_time})")
        skipped += 1
        continue

    data.append({
        "participant": participant,
        "age": age,
        "task2_time": task2_time,
        "task2_part2_time": task2_part2_time
    })
    processed += 1
    print(f"[{participant}] ✔ Processed (age={age}, t2={task2_time:.2f}s, t2p2={task2_part2_time:.2f}s)")

# -----------------------------------
# Save & plot
# -----------------------------------

df = pd.DataFrame(data)
df.to_csv("task2_age_analysis.csv", index=False)

# Only plot if we have data
if not df.empty:
    def plot_regression(x, y, label):
        model = LinearRegression()
        x_ = np.array(x).reshape(-1, 1)
        y_ = np.array(y)
        model.fit(x_, y_)
        pred = model.predict(x_)
        plt.scatter(x, y, label=f"Data: {label}")
        plt.plot(x, pred, linestyle="--", label=f"Fit: {label}")

    plt.figure(figsize=(10, 6))
    plot_regression(df["age"], df["task2_time"], "Task 2")
    plot_regression(df["age"], df["task2_part2_time"], "Task 2 Part 2")
    plt.xlabel("Age")
    plt.ylabel("Time Spent (s)")
    plt.title("Linear Regression: Age vs. Task 2 Duration")
    plt.legend()
    plt.tight_layout()
    plt.savefig("regression_task2_age.png")
    plt.show()
else:
    print("No valid data to plot.")

print("—— Summary ——")
print(f"Participants processed: {processed}")
print(f"Participants skipped:   {skipped}")
print("Saved: task2_age_analysis.csv and (if data) regression_task2_age.png")

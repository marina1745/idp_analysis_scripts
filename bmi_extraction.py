"""
Task: In Task 4 (food preference task), analyze which foods disappear first
and relate this to participant BMI.

Steps per participant:
1. Read participant metadata (gender, height, weight) from final_result*.json
2. Compute BMI.
3. From Task 4 eye-tracking CSV, determine the order in which foods disappear:
   - disappearance defined as the first frame where the food's *_Position becomes NaN
   - rank all foods by disappearance time
4. Compute an "unhealthy score" based on order of disappearance
   (pizza/chicken = unhealthy, salmon/salad = healthy).
   Unhealthy score: the higher the score, the higher the preference for unhealthy food
   score calculation:
   (4 - pizza rank) + (4 - chicken rank)
   e.g. if they rank pizza and chicken as the highest (rank 1 and 2), they get a maximum score of  5
   if they rank pizza high (rank 1) and chicken low (rank 4), they get an unhealthy score of 3
   if they rank both the lowest (rank 3 & 4), they get unhealthy score of 1

Data quality rules:
- Skip participant if required files are missing.
- Skip if JSON parsing fails.
- Skip if Task 4 CSV is missing or cannot be parsed.
- Skip if the first 50 frames of Task 4 CSV have GazeStatus == "INVALID".

Outputs:
- bmi_food_preferences.csv with columns:
    participant, gender, height_cm, weight_kg, BMI, food_order, unhealthy_score
- Console summary of processed vs skipped participants with reasons.
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

# === CONFIGURATION ===
BASE_DIR = Path("D:/Uni/WS2425/IDP/logs_filtered/logs")  # <-- CHANGE THIS if needed
TASK4_CSV_IDENTIFIER = "Task 4.csv"
JSON_IDENTIFIER = "final_result_"
FOOD_COLUMNS = {
    "pizza": "pizzaplate_Position",
    "chicken": "chickenparts_Position",
    "salmon": "smallsalmonplate_Position",
    "salad": "smallSalad_Position"
}
UNHEALTHY = {"pizza", "chicken"}
GAZE_INVALID_CHECK_N = 50

# === FUNCTIONS ===
def parse_position(pos_str):
    if pd.isna(pos_str):
        return np.nan
    try:
        _ = tuple(map(float, pos_str.strip("()").split(",")))
        return pos_str
    except:
        return np.nan

def get_bmi(height_cm, weight_kg):
    try:
        height_m = float(height_cm) / 100
        weight_kg = float(weight_kg)
        return weight_kg / (height_m ** 2)
    except:
        return np.nan

def get_disappearance_rankings(df, food_columns):
    disappearance_frame = {}
    for food, col in food_columns.items():
        if col not in df.columns:
            disappearance_frame[food] = float("inf")
            continue
        food_series = df[col].apply(parse_position)
        nan_indices = food_series[food_series.isna()].index
        disappearance_frame[food] = nan_indices[0] if len(nan_indices) > 0 else float("inf")
    sorted_foods = sorted(disappearance_frame.items(), key=lambda x: x[1])
    return [item[0] for item in sorted_foods]

def compute_unhealthy_score(food_order):
    score = 0
    for rank, food in enumerate(food_order):
        points = 4 - rank - 1  # 1st gets 3, 2nd gets 2, etc.
        if food in UNHEALTHY:
            score += points
    return score

def first_n_invalid(df: pd.DataFrame, n: int) -> bool:
    """Return True if first n frames have GazeStatus == 'INVALID'."""
    if "GazeStatus" not in df.columns:
        return False
    first = df["GazeStatus"].head(n).astype(str).str.upper()
    return len(first) > 0 and all(s == "INVALID" for s in first)

def process_experiment(exp_path: Path) -> Optional[dict]:
    json_files = list(exp_path.glob(f"{JSON_IDENTIFIER}*.json"))
    csv_files = list(exp_path.glob(f"*{TASK4_CSV_IDENTIFIER}"))
    if not json_files or not csv_files:
        print(f"[{exp_path.name}] Skipped: missing file(s)")
        return None

    # Read participant data
    try:
        with open(json_files[0], "r", encoding="utf-8") as f:
            data = json.load(f)
        answers = data["data_object5"]["answers"]
        gender = answers[1]
        height = answers[5]
        weight = answers[6]
        bmi = get_bmi(height, weight)
    except Exception as e:
        print(f"[{exp_path.name}] Skipped: JSON read error ({e})")
        return None

    # Load CSV
    try:
        df = pd.read_csv(csv_files[0], sep=";")
    except Exception as e:
        print(f"[{exp_path.name}] Skipped: CSV load error ({e})")
        return None

    # GazeStatus INVALID check
    if first_n_invalid(df, GAZE_INVALID_CHECK_N):
        print(f"[{exp_path.name}] Skipped: first {GAZE_INVALID_CHECK_N} frames INVALID in Task 4")
        return None

    food_order = get_disappearance_rankings(df, FOOD_COLUMNS)
    unhealthy_score = compute_unhealthy_score(food_order)

    return {
        "participant": exp_path.name,
        "gender": gender,
        "height_cm": height,
        "weight_kg": weight,
        "BMI": bmi,
        "food_order": food_order,
        "unhealthy_score": unhealthy_score
    }

# === MAIN ===
def main():
    results = []
    processed = 0
    skipped = 0

    for folder in sorted(BASE_DIR.glob("Exp_*")):
        result = process_experiment(folder)
        if result:
            results.append(result)
            processed += 1
        else:
            skipped += 1

    df = pd.DataFrame(results)
    df.to_csv("bmi_food_preferences.csv", index=False)

    print("✅ Analysis complete. Results saved to 'bmi_food_preferences.csv'")
    print("—— Summary ——")
    print(f"Participants processed: {processed}")
    print(f"Participants skipped:   {skipped}")

if __name__ == "__main__":
    main()

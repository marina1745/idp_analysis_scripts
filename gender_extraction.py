import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
"""
This file goes through all participants and if their data is valid (Task 1 GazeStatus not marked as INVALID)
We will see in which order they ranked the 6 objects of task 1
The result of the ranking and whether they are a man or woman is stored in object_rankings_by_gender.csv
Rank 1 means most liked object, rank 6 least liked
"""
# === CONFIGURATION ===
BASE_DIR = Path("D:/Uni/WS2425/IDP/logs_filtered/logs")  # Change this to your actual path
DROP_LAST_FRAMES = 30
AVG_OVER_FRAMES = 10

object_columns = {
    "whiskey glass": "whiskey glass_Position",
    "yoga mat": "YogaMatA3 Triangulate_Position",
    "vr headset": "VR_Headset_4_Position",
    "bbq tools": "bbq tools_Position",
    "paint box": "PaintBox_Position",
    "backpack": "backpackNeu_Position"
}

def parse_position(pos_str):
    try:
        return tuple(map(float, pos_str.strip("()").split(",")))
    except:
        return (np.nan, np.nan, np.nan)

def process_experiment_folder(exp_path: Path):
    json_candidates = list(exp_path.glob("final_result_*.json"))
    json_path = json_candidates[0] if json_candidates else None

    csv_candidates = list(exp_path.glob("*EyeTracking-Task1.csv"))
    csv_path = csv_candidates[0] if csv_candidates else None

    if not json_path or not csv_path:
        print(f"Missing files in {exp_path.name}")
        return None

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    gender = data.get("data_object5", {}).get("answers", [None, None])[1]

    try:
        df = pd.read_csv(csv_path, sep=";")
    except Exception as e:
        print(f"CSV load error in {exp_path.name}: {e}")
        return None

    # ✅ NEW CHECK: Skip participant if first 50 frames are all INVALID
    if "GazeStatus" not in df.columns:
        print(f"No GazeStatus column in {exp_path.name}")
        return None
    first50 = df["GazeStatus"].head(50).astype(str).str.upper()
    if all(status == "INVALID" for status in first50):
        print(f"Skipping {exp_path.name} (first 50 frames INVALID)")
        return None
    else:
        print(f"Processing {exp_path.name}")

    df_cleaned = df.iloc[:-DROP_LAST_FRAMES]

    final_positions = {}
    for name, col in object_columns.items():
        if col not in df_cleaned.columns:
            print(f"Missing column {col} in {exp_path.name}")
            return None
        coords = df_cleaned[col].apply(parse_position).dropna()
        last_coords = coords.iloc[-AVG_OVER_FRAMES:]
        x_coords = [pos[0] for pos in last_coords]
        final_positions[name] = np.mean(x_coords)

    ranking = sorted(final_positions.items(), key=lambda x: x[1])
    ranked_objects = {name: rank+1 for rank, (name, _) in enumerate(ranking)}

    result = {"gender": gender}
    result.update(ranked_objects)
    return result

# === MAIN ===
all_results = []
skipped = 0
processed = 0

for exp_folder in sorted(BASE_DIR.glob("Exp_*")):
    result = process_experiment_folder(exp_folder)
    if result:
        all_results.append(result)
        processed += 1
    else:
        skipped += 1

df_all = pd.DataFrame(all_results)
df_all.to_csv("object_rankings_by_gender.csv", index=False)
print("✅ Done! Results saved to: object_rankings_by_gender.csv")
print(f"Participants processed: {processed}")
print(f"Participants skipped: {skipped}")
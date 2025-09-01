"""
Task: For Task 3 (conversation task), extract each participant's:
  - specific and broad ethnicity from final_result*.json
  - the first and second NPC they spoke to (from 'Specific-Task 3.csv')

Data-quality rule:
  - Skip a participant if the first 50 frames in the Task 3 CSV have
    GazeStatus == "INVALID" (case-insensitive).
  - Also skip if required files/columns are missing or no user turns are found.

Outputs:
  - npc_ethnicity_order.csv with columns:
      Participant, Specific Ethnicity, Broad Ethnicity, First NPC, Second NPC
  - Console summary with per-participant decisions and final processed/skipped counts.
"""

import os
import json
from pathlib import Path
from typing import Optional

import pandas as pd

# -------------------------------
# Configuration
# -------------------------------
ROOT_DIR = Path(r"D:/Uni/WS2425/IDP/logs_filtered/logs")  # change if needed
GAZE_INVALID_CHECK_N = 50

# -------------------------------
# Helpers
# -------------------------------
def load_csv(csv_path: Path) -> pd.DataFrame:
    """Robust CSV loader with BOM/whitespace-stripped column names."""
    df = pd.read_csv(csv_path, sep=";", dtype=str, encoding="utf-8", on_bad_lines="skip")
    df.columns = [c.strip().replace("\ufeff", "") for c in df.columns]
    return df

def first_n_invalid(df: pd.DataFrame, n: int) -> bool:
    """True if the first n frames have GazeStatus == 'INVALID' (case-insensitive)."""
    if "GazeStatus" not in df.columns:
        return False
    first = df["GazeStatus"].head(n).astype(str).str.upper()
    return len(first) > 0 and all(s == "INVALID" for s in first)

def get_ethnicities(json_path: Path) -> tuple[Optional[str], Optional[str]]:
    """
    Extract (specific_ethnicity, broad_ethnicity) from final_result*.json.
    - specific_ethnicity = answers[3]
    - broad_ethnicity    = first element of answers[4] if it's a list
    """
    with open(json_path, "r", encoding="utf-8") as f:
        jd = json.load(f)
    answers = jd.get("data_object5", {}).get("answers", [])
    specific = answers[3] if len(answers) > 3 else None
    broad = answers[4][0] if len(answers) > 4 and isinstance(answers[4], list) and answers[4] else None
    return specific, broad

# -------------------------------
# Main
# -------------------------------
results = []
processed = 0
skipped = 0

for folder in sorted(ROOT_DIR.iterdir()):
    if not folder.is_dir() or not folder.name.startswith("Exp_"):
        continue

    participant = folder.name

    # Find files
    json_file = next((p for p in folder.glob("final_result_*.json")), None)
    specific_csv = next((p for p in folder.glob("*Specific-Task 3.csv")), None)
    eyetracking_csv = next((p for p in folder.glob("*EyeTracking-Task 3.csv")), None)

    if not json_file or not specific_csv:
        print(f"[{participant}] Skipped: missing file(s) "
              f"(JSON={'OK' if json_file else 'MISSING'}, "
              f"Specific={'OK' if specific_csv else 'MISSING'})")
        skipped += 1
        continue

    # Read JSON → ethnicities
    try:
        specific_ethnicity, broad_ethnicity = get_ethnicities(json_file)
    except Exception as e:
        print(f"[{participant}] Skipped: JSON read error ({e})")
        skipped += 1
        continue

    # Load CSV(s)
    try:
        df_specific = load_csv(specific_csv)
        df_eye = load_csv(eyetracking_csv) if eyetracking_csv else None
    except Exception as e:
        print(f"[{participant}] Skipped: CSV load error ({e})")
        skipped += 1
        continue

    # GazeStatus INVALID check — only if EyeTracking file exists
    if df_eye is not None and first_n_invalid(df_eye, GAZE_INVALID_CHECK_N):
        print(f"[{participant}] Skipped: first {GAZE_INVALID_CHECK_N} frames INVALID in EyeTracking-Task 3")
        skipped += 1
        continue

    # Normalize column names we need
    if "Role" not in df_specific.columns or "NPCName" not in df_specific.columns:
        print(f"[{participant}] Skipped: missing required columns ('Role', 'NPCName')")
        skipped += 1
        continue

    # Extract NPC sequence
    npc_sequence = df_specific[df_specific["Role"] == "user"]["NPCName"].dropna().astype(str).tolist()
    if not npc_sequence:
        print(f"[{participant}] Skipped: no user turns found")
        skipped += 1
        continue

    npc_first = None
    npc_second = None
    for npc in npc_sequence:
        if npc_first is None:
            npc_first = npc
        elif npc != npc_first:
            npc_second = npc
            break

    results.append({
        "Participant": participant,
        "Specific Ethnicity": specific_ethnicity,
        "Broad Ethnicity": broad_ethnicity,
        "First NPC": npc_first,
        "Second NPC": npc_second
    })
    processed += 1
    print(f"[{participant}] ✔ Processed (first='{npc_first}', second='{npc_second}')")

# Save results
df_results = pd.DataFrame(results)
df_results.to_csv("npc_ethnicity_order.csv", index=False)
print("Saved to npc_ethnicity_order.csv")

print("—— Summary ——")
print(f"Participants processed: {processed}")
print(f"Participants skipped:   {skipped}")

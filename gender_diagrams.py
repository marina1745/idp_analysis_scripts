import pandas as pd
from scipy.stats import mannwhitneyu
import seaborn as sns
import matplotlib.pyplot as plt

"""
This file generates the info from the report for task 1 (gender). It calculates a mann-whitney u test on the
difference on how men and women ranked the objects in task 1, it generates boxplots of the preferences of men and
women per object
This requires the extracted rankings from gender_extraction.py
"""

# === CONFIGURATION ===
FILE = "object_rankings_by_gender.csv"
OBJECTS = ["whiskey glass", "paint box", "bbq tools", "yoga mat", "vr headset", "backpack"]

# === 1. Load data ===
df = pd.read_csv(FILE)

# === 2. Group means (optional for report tables) ===
print("=== Average Rankings Per Gender ===")
print(df.groupby("gender")[OBJECTS].mean())
print()

# === 3. Mann-Whitney U Tests ===
print("=== Mann–Whitney U Test Results ===")
report_lines = []
for obj in OBJECTS:
    group1 = df[df["gender"] == "Man"][obj]
    group2 = df[df["gender"] == "Woman"][obj]

    if len(group1) == 0 or len(group2) == 0:
        print(f"{obj}: Not enough data")
        continue

    stat, p = mannwhitneyu(group1, group2, alternative='two-sided')
    significance = "✔️ significant" if p < 0.05 else "✖️ not significant"
    line = f"{obj}: p = {p:.4f} → {significance}"
    print(line)
    report_lines.append(line)

# === 4. Boxplots ===
print("\nGenerating boxplots...")
for obj in OBJECTS:
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df, x="gender", y=obj)
    plt.title(f"Ranking of {obj} by Gender")
    plt.ylabel("Rank (lower = more left)")
    plt.xlabel("Gender")
    plt.tight_layout()
    plt.show()

# === 5. Summary for your report ===
print("\n=== Report Summary Text ===")
print("We used the Mann–Whitney U test to compare object rankings between male and female participants.")
print("The following results were observed:\n")
for line in report_lines:
    print("• " + line)
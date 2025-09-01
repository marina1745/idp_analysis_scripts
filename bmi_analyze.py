import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



"""
Task: Analyze the relationship between participant BMI and unhealthy food
preferences (based on results from bmi_extraction.py).

Inputs:
- bmi_food_preferences.csv (output of healthtask.py), containing for each
  participant:
    * gender
    * height_cm, weight_kg, BMI
    * food_order (list of disappearance order)
    * unhealthy_score (numeric)

Steps:
1. Load results and drop rows with missing BMI or unhealthy_score.
2. Plot scatter of BMI vs. unhealthy_score, colored by gender.
3. Categorize participants into BMI groups:
     - Underweight (<18.5)
     - Healthy (18.5–24.9)
     - Overweight (25–29.9)
     - Obese (≥30)
4. Print counts of participants per BMI category.
5. Visualize unhealthy scores by BMI category:
     - Strip plot by gender
     - Combined boxplot + stripplot with gender colors
6. Plot overall bar chart of participant counts per BMI category.

Outputs:
- bmi_vs_unhealthy_score.png : scatterplot (BMI vs unhealthy score, gender hue)
- unhealthy_score_stripplot.png : stripplot of unhealthy score by BMI group
- bmi_box_strip_combo.png : combined box + stripplot by BMI group/gender
- bmi_category_counts.png : bar chart of counts by BMI group
- bmi_violin_gender.png: gender split for bmi + unhealthy score
- Console printout: counts of participants in each BMI category
"""

# === Load the data ===
df = pd.read_csv("bmi_food_preferences.csv")

# === Drop missing BMI rows just in case ===
df = df.dropna(subset=["BMI", "unhealthy_score"])




# === 1. Plot BMI vs Unhealthy Preference Score ===
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="BMI", y="unhealthy_score", hue="gender")
plt.title("BMI vs Unhealthy Food Preference Score")
plt.xlabel("BMI")
plt.ylabel("Unhealthy Preference Score")
plt.grid(True)
plt.tight_layout()
plt.savefig("bmi_vs_unhealthy_score.png")
plt.show()


# efebozkir@gmail.com
# === 2. Categorize BMI ===
def categorize_bmi(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Healthy"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obese"

df["bmi_category"] = df["BMI"].apply(categorize_bmi)

# === 3. Count categories ===
bmi_counts = df["bmi_category"].value_counts().sort_index()
print("=== BMI Category Counts ===")
print(bmi_counts)



sns.stripplot(data=df, x="bmi_category", y="unhealthy_score", hue="gender", dodge=True, jitter=0.25)
plt.title("Unhealthy Preference Score by BMI Group")
plt.xlabel("BMI Category")
plt.ylabel("Unhealthy Preference Score")
plt.tight_layout()
plt.savefig("unhealthy_score_stripplot.png")
plt.show()

plt.figure(figsize=(8, 5))

sns.boxplot(data=df, x="bmi_category", y="unhealthy_score",  color="lightgray", order=["Underweight", "Healthy", "Overweight", "Obese"], width=0.5)
sns.stripplot(data=df, x="bmi_category", y="unhealthy_score", hue="gender", dodge=True, jitter=0.25, alpha=0.6, order=["Underweight", "Healthy", "Overweight", "Obese"],  palette={"Man": "#005bd1", "Woman": "#f50595"} )

plt.title("Unhealthy Preference Score by BMI Group")
plt.xlabel("BMI Category")
plt.ylabel("Unhealthy Preference Score")
plt.legend(title="Gender", loc="lower right")
plt.tight_layout()
plt.savefig("bmi_box_strip_combo.png")
plt.show()

# === 4. Bar plot of BMI categories ===
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="bmi_category", order=["Underweight", "Healthy", "Overweight", "Obese"])
plt.title("Participant Count by BMI Category")
plt.xlabel("BMI Category")
plt.ylabel("Number of Participants")
plt.tight_layout()
plt.savefig("bmi_category_counts.png")
plt.show()


plt.figure(figsize=(9, 5))

order = ["Underweight", "Healthy", "Overweight", "Obese"]

# Violin plot by gender
sns.violinplot(
    data=df, x="bmi_category", y="unhealthy_score",
    hue="gender", order=order, cut=0, inner="quartile", split=True,
    palette={"Man": "#005bd1", "Woman": "#f50595"}
)

# Overlay stripplot (individual points)
sns.stripplot(
    data=df, x="bmi_category", y="unhealthy_score",
    hue="gender", dodge=True, order=order,
    palette={"Man": "#005bd1", "Woman": "#f50595"},
    size=5, alpha=0.8, jitter=True
)

# Fix legend (avoid duplicates from hue in both plots)
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles[0:2], labels[0:2], title="Gender",
           loc="upper left", bbox_to_anchor=(1.02, 1))

plt.title("Unhealthy Preference Score by BMI Group (Violin + Points)")
plt.xlabel("BMI Category")
plt.ylabel("Unhealthy Preference Score")
plt.tight_layout()
plt.savefig("bmi_violin_points.png", bbox_inches="tight")
plt.show()


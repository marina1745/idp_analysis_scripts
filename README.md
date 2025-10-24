# IDP Analysis Scripts – Privacy VR Project

This repository contains the **data analysis and visualization scripts** developed for the *Privacy VR* project — a research study investigating how personal attributes can be inferred from **eye-tracking data** in immersive virtual environments.  
All analyses were conducted on gaze and behavioral data collected in Unity using the **Varjo XR-3 headset**.

---

## 🧠 Research Overview

The *Privacy VR* project examines the privacy risks of eye-tracking in future XR applications by predicting personal user attributes such as **age**, **gender**, **BMI**, and **ethnic background** purely from gaze patterns and object interactions.

Participants completed several task-oriented VR scenarios (e.g., *gender recognition*, *food preference*, *age reconstruction*), during which detailed gaze, position, and task metrics were logged.  
This repository contains the full **post-processing and statistical analysis pipeline** used to evaluate that data.

---
This project was implemented in **Python 3.10+**.  
Install all required libraries with:

```bash
pip install pandas numpy scipy matplotlib seaborn statsmodels
````
## 🔍 Usage Instructions

Place all participant log files in a folder named data/ at the repository root.

Run the extraction scripts first to preprocess data:
```bash
python gender_extraction.py
python race_extraction.py
python bmi_extraction.py
```

Execute the corresponding analysis scripts:
```bash
python gender_diagrams.py
python race_analysis.py
python bmi_analyze.py
python age_object_difference_analysis.py
python age_time_spent_analysis.py
```

Generated plots and summary tables will appear in a folder named figures/ (create it if not present).

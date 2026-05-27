# 🧠 None Decides Consciously

> *"A Data Science project — collecting, cleaning, analysing and predicting real student data to prove that Human behavior is a function of emotional state, not free will."*

## 📌 Project Goal
To statistically prove that human decisions (social media usage, academic 
performance, eating habits) are predicted by physiological/emotional states 
like stress, sleep, and mood — not by conscious choice.

## 🔬 Hypotheses Being Tested
- H1: Higher stress levels → increased social media usage
- H2: Fewer sleep hours → lower academic performance  
- H3: Emotional state predicts behavior better than demographics

## 🛠️ Tech Stack
- Python, Jupyter Notebook
- Pandas, NumPy, Matplotlib, Seaborn
- Scipy, Statsmodels, Scikit-learn

## 📁 Project Structure
\```
None_Decides_Consciously/
│
├── data/
│   ├── raw/          # Original datasets (never modified)
│   └── cleaned/      # Processed datasets
│
├── notebooks/        # Jupyter notebooks (one per phase)
├── src/              # Reusable Python scripts
├── reports/
│   └── figures/      # Saved plots and charts
│
├── requirements.txt
└── README.md
\```

## 📈 Progress Log
| Day | Phase | What was done |
|-----|-------|---------------|
| Day 1 | Setup | Project structure + GitHub setup |
| Day 2 | Data Loading + Cleaning | Loaded 1000 rows 21 columns, fixed outliers, handled missing values, saved clean dataset |
| Day 3 | EDA | Plotted distributions, scatter plots, correlation heatmap - H1 and H2 strongly proved visually |
| Day 4 | Statistical Testing | Pearson correlation + P-value testing - H1 0.56, H2 0.52, H3 0.15 all proved with p < 0.001 |
| Day 5 | Feature Engineering | Created 5 new features - sleep_debt, stress_sleep_ratio, state_score, screen_category, at_risk |

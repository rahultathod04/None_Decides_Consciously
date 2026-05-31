import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ================================
# PAGE CONFIG
# ================================
st.set_page_config(
    page_title="EmoDriven",
    page_icon="🧠",
    layout="centered"
)

# ================================
# LOAD DATA AND TRAIN MODELS
# ================================
@st.cache_data
def load_and_train():
    df = pd.read_csv('data/cleaned/student_lifestyle_final.csv')

    features = ['stress_level', 'anxiety_score', 'sleep_hours',
                'mood_rating', 'sleep_debt', 'stress_sleep_ratio',
                'state_score']

    # AT RISK classifier
    X = df[features]
    y = df['at_risk']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    return df, clf, features

df, clf, features = load_and_train()

# ================================
# RECOMMENDATION FUNCTION
# ================================
def recommend(student_type):
    recs = {
        'Burned Out': {
            'priority' : '🔴 urgent — act now',
            'problem'  : 'too much stress + not enough sleep',
            'steps'    : [
                'sleep → try to get at least 7 hours tonight',
                'stress → 10 min deep breathing before bed',
                'social media → max 1 hour per day',
                'screens → no phone 1 hour before sleep'
            ],
            'expected' : 'you will feel better in 3 to 4 weeks'
        },
        'Struggling': {
            'priority' : '🟡 moderate — small changes needed',
            'problem'  : 'sleep is inconsistent, stress is building',
            'steps'    : [
                'sleep → fix a sleep time and stick to it daily',
                'stress → short walk 3 times a week',
                'social media → cut down by 30 min each day',
                'study → add 30 min of focused study daily'
            ],
            'expected' : 'things will improve in 2 to 3 weeks'
        },
        'Thriving': {
            'priority' : '🟢 good — keep going',
            'problem'  : 'no major problems found',
            'steps'    : [
                'sleep → keep sleeping 7 or more hours',
                'stress → whatever you are doing, keep doing it',
                'social media → your usage is already healthy',
                'study → push yourself a little more now'
            ],
            'expected' : 'you are in the top 24% — maintain it'
        }
    }
    return recs[student_type]

# ================================
# APP HEADER
# ================================
st.title("🧠 EmoDriven")
st.markdown(
    "*Proving that human behavior is driven by emotional state — not free will*"
)
st.divider()

# ================================
# SIDEBAR — USER INPUT
# ================================
st.sidebar.header("📋 Enter Your Details")

stress  = st.sidebar.slider("Stress Level",   1, 10, 5)
anxiety = st.sidebar.slider("Anxiety Score",  1, 10, 5)
sleep   = st.sidebar.slider("Sleep Hours",    3, 10, 6)
mood    = st.sidebar.slider("Mood Rating",    1, 10, 5)

# Calculate engineered features
sleep_debt         = 8 - sleep
stress_sleep_ratio = stress / sleep
state_score        = (stress + anxiety + (10 - mood)) / 3

# ================================
# CLASSIFY STUDENT TYPE
# ================================
if stress >= 7 and sleep <= 5.5:
    student_type = 'Burned Out'
elif stress <= 4.5 and sleep >= 7:
    student_type = 'Thriving'
else:
    student_type = 'Struggling'

# AT RISK prediction
input_data = pd.DataFrame([[stress, anxiety, sleep, mood,
                             sleep_debt, stress_sleep_ratio,
                             state_score]], columns=features)
at_risk = clf.predict(input_data)[0]

# ================================
# RESULTS SECTION
# ================================
col1, col2, col3 = st.columns(3)
col1.metric("Sleep Debt",   f"{sleep_debt:.1f} hrs")
col2.metric("State Score",  f"{state_score:.1f} / 10")
col3.metric("At Risk",      "Yes 🔴" if at_risk else "No 🟢")

st.divider()

# Student type
colors = {
    'Burned Out' : '🔴',
    'Struggling' : '🟡',
    'Thriving'   : '🟢'
}
st.subheader(f"{colors[student_type]} You are : {student_type}")

# Recommendation
rec = recommend(student_type)
st.caption(rec['priority'])
st.write(f"**Biggest problem:** {rec['problem']}")

st.subheader("📋 Your Action Plan")
for i, step in enumerate(rec['steps'], 1):
    st.write(f"**Step {i}:** {step}")

st.info(f"⏱️ {rec['expected']}")

st.divider()

# ================================
# DATASET INSIGHTS
# ================================
st.subheader("📊 How You Compare to 1000 Students")

fig, axes = plt.subplots(1, 2, figsize=(10, 3))

# Stress comparison
axes[0].hist(df['stress_level'], bins=10,
             color='lightcoral', edgecolor='white')
axes[0].axvline(x=stress, color='black',
                linestyle='--', linewidth=2,
                label=f'You ({stress})')
axes[0].set_title('Stress Level')
axes[0].legend()

# Sleep comparison
axes[1].hist(df['sleep_hours'], bins=10,
             color='steelblue', edgecolor='white')
axes[1].axvline(x=sleep, color='black',
                linestyle='--', linewidth=2,
                label=f'You ({sleep} hrs)')
axes[1].set_title('Sleep Hours')
axes[1].legend()

plt.tight_layout()
st.pyplot(fig)

st.divider()
st.caption("EmoDriven — built by Rahul Tathod | Data Science Project")
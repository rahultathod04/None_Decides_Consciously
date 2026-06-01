import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import anthropic
from dotenv import load_dotenv
import os

# Load API key from .env
load_dotenv()

# ================================
# PAGE CONFIG
# ================================
st.set_page_config(
    page_title="EmoDriven",
    page_icon="🧠",
    layout="centered"
)

# ================================
# LOAD DATA AND TRAIN MODEL
# ================================
@st.cache_data
def load_and_train():
    df = pd.read_csv('data/cleaned/student_lifestyle_final.csv')

    features = ['stress_level', 'anxiety_score', 'sleep_hours',
                'mood_rating', 'sleep_debt', 'stress_sleep_ratio',
                'state_score']

    X = df[features]
    y = df['at_risk']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    return df, clf, features

df, clf, features = load_and_train()

# ================================
# CLASSIFY FUNCTION
# ================================
def get_student_type(stress, sleep):
    if stress >= 7 and sleep <= 5.5:
        return 'Burned Out'
    elif stress <= 4.5 and sleep >= 7:
        return 'Thriving'
    else:
        return 'Struggling'

# ================================
# APP HEADER
# ================================
st.title("🧠 EmoDriven")
st.markdown(
    "*proving that human behavior is driven by emotional state — not free will*"
)
st.divider()

# ================================
# SIDEBAR SLIDERS
# ================================
st.sidebar.header("📋 enter your details")
stress  = st.sidebar.slider("stress level",  1, 10, 5)
anxiety = st.sidebar.slider("anxiety score", 1, 10, 5)
sleep   = st.sidebar.slider("sleep hours",   3, 10, 6)
mood    = st.sidebar.slider("mood rating",   1, 10, 5)

# Calculate features
sleep_debt         = 8 - sleep
stress_sleep_ratio = stress / sleep
state_score        = (stress + anxiety + (10 - mood)) / 3
student_type       = get_student_type(stress, sleep)

# AT RISK prediction
input_data = pd.DataFrame([[stress, anxiety, sleep, mood,
                             sleep_debt, stress_sleep_ratio,
                             state_score]], columns=features)
at_risk = clf.predict(input_data)[0]

# ================================
# METRICS
# ================================
col1, col2, col3 = st.columns(3)
col1.metric("sleep debt",   f"{sleep_debt:.1f} hrs")
col2.metric("state score",  f"{state_score:.1f} / 10")
col3.metric("at risk",      "yes 🔴" if at_risk else "no 🟢")

st.divider()

# ================================
# STUDENT TYPE
# ================================
icons = {'Burned Out': '🔴', 'Struggling': '🟡', 'Thriving': '🟢'}
st.subheader(f"{icons[student_type]} you are : {student_type.lower()}")

# ================================
# AI COUNSELOR BUTTON
# ================================
st.divider()
st.subheader("🤖 your personal ai counselor")
st.caption("get a unique message written just for your exact situation")

if st.button("get my personalised advice"):
    with st.spinner("analysing your emotional state..."):
        try:
            from groq import Groq

            client = Groq(api_key=os.getenv("GROQ_API_KEY"))

            prompt = f"""
You are a student wellbeing counselor.
A student has shared their emotional state with you.

Their values:
- stress level     : {stress} out of 10
- anxiety score    : {anxiety} out of 10
- sleep hours      : {sleep} hours per night
- mood rating      : {mood} out of 10
- sleep debt       : {sleep_debt:.1f} hours
- state score      : {state_score:.1f} out of 10
- student type     : {student_type}
- at academic risk : {"yes" if at_risk else "no"}

Write a short warm honest message directly to this student.
- address their exact numbers and situation
- tell them what their numbers actually mean
- give 2 specific things they can do today
- be encouraging but realistic
- write like a caring friend not a robot
- use simple everyday english, small letters
- keep it under 120 words
"""

            response = client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300
            )

            st.success(response.choices[0].message.content)

        except Exception as e:
            st.error(f"something went wrong: {e}")
           try:
    from groq import Groq

    client = Groq(
        api_key=os.getenv("GROQ_API_KEY")
    )

    prompt = f"""
You are a student wellbeing counselor.
A student has shared their emotional state with you.

Their values:
- stress level     : {stress} out of 10
- anxiety score    : {anxiety} out of 10
- sleep hours      : {sleep} hours per night
- mood rating      : {mood} out of 10
- sleep debt       : {sleep_debt:.1f} hours
- state score      : {state_score:.1f} out of 10
- student type     : {student_type}
- at academic risk : {"yes" if at_risk else "no"}

Write a short warm honest message directly to this student.
- address their exact numbers and situation
- tell them what their numbers actually mean for their life
- give 2 very specific things they can do today
- be encouraging but realistic
- write like a caring friend not a robot
- use simple everyday english, small letters
- keep it under 120 words
"""

    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300
    )

    st.success(response.choices[0].message.content)

except Exception as e:
    st.error(f"something went wrong: {e}")


            prompt = f"""
You are a student wellbeing counselor.
A student has shared their emotional state with you.

Their values:
- stress level     : {stress} out of 10
- anxiety score    : {anxiety} out of 10
- sleep hours      : {sleep} hours per night
- mood rating      : {mood} out of 10
- sleep debt       : {sleep_debt:.1f} hours
- state score      : {state_score:.1f} out of 10
- student type     : {student_type}
- at academic risk : {"yes" if at_risk else "no"}

Write a short warm honest message directly to this student.
- address their exact numbers and situation
- tell them what their numbers actually mean for their life
- give 2 very specific things they can do today
- be encouraging but realistic
- write like a caring friend not a robot
- use simple everyday english, small letters
- keep it under 120 words
"""

            message = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}]
            )

            st.success(message.content[0].text)

        except Exception as e:
            st.error(f"something went wrong: {e}")

# ================================
# COMPARISON CHARTS
# ================================
st.divider()
st.subheader("📊 how you compare to 1000 students")

fig, axes = plt.subplots(1, 2, figsize=(10, 3))

axes[0].hist(df['stress_level'], bins=10,
             color='lightcoral', edgecolor='white')
axes[0].axvline(x=stress, color='black', linestyle='--',
                linewidth=2, label=f'you ({stress})')
axes[0].set_title('stress level')
axes[0].legend()

axes[1].hist(df['sleep_hours'], bins=10,
             color='steelblue', edgecolor='white')
axes[1].axvline(x=sleep, color='black', linestyle='--',
                linewidth=2, label=f'you ({sleep} hrs)')
axes[1].set_title('sleep hours')
axes[1].legend()

plt.tight_layout()
st.pyplot(fig)

st.divider()
st.caption("eModriven — built by rahul tathod | data science project")
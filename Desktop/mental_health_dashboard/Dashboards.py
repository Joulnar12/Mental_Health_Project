# Mental Health Dashboard - Streamlit Interactive App

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import os

st.set_page_config(page_title="Mental Health Dashboard", layout="wide")
st.title("🧠 Mental Health & Lifestyle Dashboard")

# === LOAD DATA ===
def load_data():
    dataset_1 = "Mental_Health_Lifestyle_Dataset.csv"
    dataset_2 = "healthy_lifestyle_city_2021.csv"


    if not os.path.isfile(dataset_1):
        st.error(f"File not found: {dataset_1}. Please ensure it's in the working directory.")
        return None, None
    if not os.path.isfile(dataset_2):
        st.error(f"File not found: {dataset_2}. Please ensure it's in the working directory.")
        return None, None

    df1 = pd.read_csv(dataset_1)
    df2 = pd.read_csv(dataset_2)
    return df1, df2

participants, cities = load_data()
if participants is not None and cities is not None:
    st.sidebar.header("Dataset Filters")

    # === Clean Participant Data ===
    df = participants.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace(r'[()/%]', '', regex=True)
    df['stress_score'] = df['stress_level'].str.title().map({'Low': 1, 'Moderate': 2, 'High': 3})
    df['exercise_score'] = df['exercise_level'].map({'Low': 1, 'Moderate': 2, 'High': 3})
    df['has_condition'] = df['mental_health_condition'].notnull().astype(int)
    df['happiness_score'] = pd.to_numeric(df['happiness_score'], errors='coerce')

    st.subheader("📈 Correlation - Participant Data")
    corr = df[['sleep_hours', 'work_hours_per_week', 'screen_time_per_day_hours',
               'social_interaction_score', 'happiness_score', 'stress_score']].corr()
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax1)
    st.pyplot(fig1)

    st.subheader("🌟 Top 10 Happiest Individuals")
    happiest = df.sort_values(by='happiness_score', ascending=False).head(10)
    st.dataframe(happiest[['country','gender','exercise_level','sleep_hours','stress_level','happiness_score']])

    st.subheader("😞 Bottom 10 Saddest Individuals")
    saddest = df.sort_values(by='happiness_score', ascending=True).head(10)
    st.dataframe(saddest[['country','gender','exercise_level','sleep_hours','stress_level','happiness_score']])

    # === Clean City Data ===
    city_df = cities.copy()
    city_df.columns = city_df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace(r'[()/%]', '', regex=True)
    city_df['happiness'] = pd.to_numeric(city_df['happiness_levelscountry'], errors='coerce')
    city_df['gym_cost'] = pd.to_numeric(city_df['cost_of_a_monthly_gym_membershipcity'].str.replace('£',''), errors='coerce')
    city_df['pollution_index'] = pd.to_numeric(city_df['pollutionindex_score_city'], errors='coerce')
    city_df['work_hours'] = pd.to_numeric(city_df['annual_avg._hours_worked'], errors='coerce')

    # Create missing obesity_rate column if needed
    if 'obesity_levelscountry' in city_df.columns:
        city_df['obesity_rate'] = pd.to_numeric(city_df['obesity_levelscountry'].str.replace('%', ''), errors='coerce')
    else:
        city_df['obesity_rate'] = None

    st.subheader("🌆 Correlation - City-Level Factors")
    features = ['obesity_rate', 'pollution_index', 'gym_cost', 'work_hours', 'happiness']
    for f in features:
        if f in city_df.columns:
            city_df[f] = pd.to_numeric(city_df[f], errors='coerce')

    corr_city = city_df[features].dropna().corr()
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    sns.heatmap(corr_city, annot=True, cmap='YlGnBu', ax=ax2)
    st.pyplot(fig2)

    st.subheader("🏙️ Top 5 Happiest Cities")
    st.dataframe(city_df[['city', 'happiness']].sort_values(by='happiness', ascending=False).head(5))
else:
    st.stop()

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler

# === LOAD CLEANED DATA ===
df = pd.read_csv("/Users/joulnarabouchakra/Desktop/Mental_Health_Project/Mental_Health_Lifestyle_Dataset.csv")
env_df = pd.read_csv("/Users/joulnarabouchakra/Desktop/Mental_Health_Project/healthy_lifestyle_city_2021.csv")

# === CLEANING & TRANSFORMATIONS ===
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace(r'[()/%]', '', regex=True)
df['age_group'] = pd.cut(df['age'], bins=[0, 18, 25, 35, 50, 100], labels=["<18", "18–25", "26–35", "36–50", "50+"])
df['stress_level'] = df['stress_level'].str.strip().str.title()
df['stress_score'] = df['stress_level'].map({'Low': 1, 'Moderate': 2, 'High': 3})
df['exercise_score'] = df['exercise_level'].map({'Low': 1, 'Moderate': 2, 'High': 3})
df['has_condition'] = df['mental_health_condition'].notnull().astype(int)
df['happiness_score'] = pd.to_numeric(df['happiness_score'], errors='coerce')

# === FIX for ENVIRONMENTAL DF ===
env_df.columns = env_df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace(r'[()/%]', '', regex=True)
env_df['pollution_index'] = pd.to_numeric(env_df['pollutionindex_score_city'], errors='coerce')
env_df['happiness'] = pd.to_numeric(env_df['happiness_levelscountry'], errors='coerce')
env_df['gym_cost'] = pd.to_numeric(env_df['cost_of_a_monthly_gym_membershipcity'].str.replace('£', ''), errors='coerce')

# === SIDEBAR FILTERS ===
st.set_page_config(layout="wide")
st.sidebar.header("🔍 Global Filters")

gender = st.sidebar.multiselect("Gender", df["gender"].unique(), default=df["gender"].unique())
country = st.sidebar.multiselect("Country", df["country"].unique(), default=df["country"].unique())
age_options = df["age_group"].astype(str).unique().tolist()
age_group = st.sidebar.multiselect("Age Group", options=age_options, default=age_options)

df_filt = df[(df["gender"].isin(gender)) & (df["country"].isin(country)) & (df["age_group"].astype(str).isin(age_group))]

# === KPIs ===
st.markdown("""
    <style>
.kpi-box {
    background-color: #f0f2f6;
    padding: 1.2rem;
    border-radius: 10px;
    text-align: center;
    box-shadow: 1px 1px 5px rgba(0,0,0,0.1);
    margin-bottom: 20px;
    min-height: 130px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}
.kpi-title {
    color: #222;
    font-weight: 600;
    font-size: 1.1rem;
}
.kpi-value {
    font-weight: bold;
    color: #005b96;
    font-size: 1.8rem;
    line-height: 1.8rem;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
    <h1>🧠 Mental Health & Lifestyle Dashboard</h1>
""", unsafe_allow_html=True)

colkpi1, colkpi2, colkpi3 = st.columns(3)
with colkpi1:
    pct_high_stress = (df_filt['stress_level'] == 'High').mean() * 100
    st.markdown(f"""<div class='kpi-box'><div class='kpi-title'>🔴 High Stress %</div><div class='kpi-value'>{pct_high_stress:.1f}%</div></div>""", unsafe_allow_html=True)
with colkpi2:
    pct_youth_high_stress = df_filt[(df_filt['age_group'] == '18–25') & (df_filt['stress_level'] == 'High')].shape[0] / df_filt[df_filt['age_group'] == '18–25'].shape[0] * 100
    st.markdown(f"""<div class='kpi-box'><div class='kpi-title'>👥 18–25 with High Stress</div><div class='kpi-value'>{pct_youth_high_stress:.1f}%</div></div>""", unsafe_allow_html=True)
with colkpi3:
    pct_screen_mental = df[(df['screen_time_per_day_hours'] >= 5) & (df['has_condition'] == 1)].shape[0] / df[df['screen_time_per_day_hours'] >= 5].shape[0] * 100
    st.markdown(f"""<div class='kpi-box'><div class='kpi-title'>📱 5h+ Screen w/ Condition</div><div class='kpi-value'>{pct_screen_mental:.1f}%</div></div>""", unsafe_allow_html=True)

# === Section A ===
st.markdown("<h2>📊 Section A — Lifestyle & Demographic Insights</h2>", unsafe_allow_html=True)

colA1, colA2, colA3 = st.columns(3)
with colA1:
    gender_data = df_filt.groupby("gender").agg({"happiness_score": "mean", "stress_score": "mean"}).reset_index()
    fig = px.pie(
        gender_data,
        names='gender',
        values='happiness_score',
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Set3,
        title="Happiness Score by Gender"
    )
    fig.update_layout(title_font_size=18)
    st.plotly_chart(fig, use_container_width=True, height=300)
with colA2:
    diet_stats = df_filt.groupby("diet_type")[["happiness_score", "stress_score"]].mean().reset_index()
    fig = px.bar(diet_stats, x="diet_type", y=["happiness_score", "stress_score"], barmode="group", color_discrete_sequence=px.colors.qualitative.Pastel1)
    st.plotly_chart(fig, use_container_width=True)
with colA3:
    scaler = MinMaxScaler()
    df_filt["lifestyle_score"] = scaler.fit_transform(df_filt[["sleep_hours", "exercise_score", "social_interaction_score"]].fillna(0)).mean(axis=1)
    df_filt["lifestyle_bin"] = pd.cut(df_filt["lifestyle_score"], bins=5).astype(str)
    lifestyle_avg = df_filt.groupby("lifestyle_bin", observed=True)["happiness_score"].mean().reset_index()
    fig = px.line(lifestyle_avg, x="lifestyle_bin", y="happiness_score", markers=True, title="Avg Happiness by Lifestyle Group", color_discrete_sequence=px.colors.qualitative.Bold)
    st.plotly_chart(fig, use_container_width=True)

# === Profile Definition for Section B Charts ===
happiest = df_filt.sort_values(by="happiness_score", ascending=False).head(10)
saddest = df_filt.sort_values(by="happiness_score", ascending=True).head(10)

# === Section B ===
st.markdown("<h2>🌍 Section B — Environmental Insights</h2>", unsafe_allow_html=True)

colB1, colB2, colB3 = st.columns(3)
with colB1:
    pollution_happiness = env_df.dropna(subset=["pollution_index", "happiness"])
    fig = px.scatter(pollution_happiness, x="pollution_index", y="happiness", color="pollution_index",
                     trendline="ols", title="Pollution Index vs Happiness")
    st.plotly_chart(fig, use_container_width=True, height=300)
with colB2:
    categories = ["sleep_hours", "work_hours_per_week", "screen_time_per_day_hours", "social_interaction_score", "stress_score"]
    happy_avg = happiest[categories].mean().reset_index()
    sad_avg = saddest[categories].mean().reset_index()

    fig = px.bar(
        pd.concat([
            happy_avg.rename(columns={0: "value"}).assign(Group="Happiest"),
            sad_avg.rename(columns={0: "value"}).assign(Group="Saddest")
        ]),
        x="index", y="value", color="Group", barmode="group",
        title="Happiest vs Saddest Lifestyle Comparison",
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    st.plotly_chart(fig, use_container_width=True, height=300)
with colB3:
    env_df['bottle_water_cost'] = pd.to_numeric(env_df['cost_of_a_bottle_of_watercity'].str.replace('£', ''), errors='coerce')
    bottle_happy = env_df.dropna(subset=["bottle_water_cost", "happiness"])
    fig = px.scatter(
        bottle_happy,
        x="bottle_water_cost",
        y="happiness",
        color="bottle_water_cost",
        color_continuous_scale=px.colors.sequential.Turbo,
        trendline="ols",
        title="Bottle Water Cost vs Happiness"
    )
    st.plotly_chart(fig, use_container_width=True, height=300)

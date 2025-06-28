import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# === 1. LOAD & CLEAN DATA ===
df = pd.read_csv("Mental_Health_Lifestyle_Dataset.csv")
df.columns = (
    df.columns
      .str.strip()
      .str.lower()
      .str.replace(" ", "_")
      .str.replace(r"[()/%]", "", regex=True)
)

# === 2. FEATURE ENGINEERING ===
level_map = {"Low": 1, "Moderate": 2, "High": 3}
df["stress_score"]   = df["stress_level"].str.strip().map(level_map)
df["exercise_score"] = df["exercise_level"].str.strip().map(level_map)
df["happiness_score"] = pd.to_numeric(df["happiness_score"], errors="coerce")
df["has_condition"]   = df["mental_health_condition"].notnull().astype(int)

# map diet_type → diet_score
diet_map = {
    "Junk Food":   1,
    "Keto":        2,
    "Balanced":    3,
    "Vegetarian":  4,
    "Vegan":       5
}
df["diet_score"] = df["diet_type"].map(diet_map)

# drop NA for analyses
vars_all = [
    "sleep_hours", "screen_time_per_day_hours", "work_hours_per_week",
    "social_interaction_score", "exercise_score", "diet_score",
    "has_condition"
]
df_corr = df[vars_all + ["happiness_score", "stress_score"]].dropna()

# === 3. CORRELATION HEATMAPS ===

# full matrix: health factors vs. happiness
plt.figure(figsize=(8, 6))
sns.heatmap(
    df_corr.corr().loc[vars_all + ["happiness_score"], vars_all + ["happiness_score"]],
    annot=True, cmap="coolwarm", fmt=".2f"
)
plt.title("Correlation Matrix (Health Factors vs Happiness Score)")
plt.tight_layout()
plt.show()

# full matrix: health factors vs. stress
plt.figure(figsize=(8, 6))
sns.heatmap(
    df_corr.corr().loc[vars_all + ["stress_score"], vars_all + ["stress_score"]],
    annot=True, cmap="coolwarm", fmt=".2f"
)
plt.title("Correlation Matrix (Health Factors vs Stress Level)")
plt.tight_layout()
plt.show()

# correlation with happiness only
plt.figure(figsize=(4, 4))
sns.heatmap(
    df_corr[vars_all + ["happiness_score"]].corr()[["happiness_score"]],
    annot=True, cmap="Greens", fmt=".2f"
)
plt.title("Correlation with Happiness Score")
plt.tight_layout()
plt.show()

# correlation with stress only
plt.figure(figsize=(4, 4))
sns.heatmap(
    df_corr[vars_all + ["stress_score"]].corr()[["stress_score"]],
    annot=True, cmap="Reds", fmt=".2f"
)
plt.title("Correlation with Stress Score")
plt.tight_layout()
plt.show()

# === 4. CATEGORICAL & DISTRIBUTION PLOTS ===

# happiness by diet
plt.figure(figsize=(6, 4))
sns.boxplot(data=df, x="diet_type", y="happiness_score")
plt.title("Happiness Score by Diet Type")
plt.tight_layout()
plt.show()

# stress level by exercise level
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="exercise_level", hue="stress_level")
plt.title("Stress Level by Exercise Level")
plt.tight_layout()
plt.show()

# === 5. GROUPED BAR PLOTS ===

# country profile
country_avg = (
    df.groupby("country")[vars_all + ["happiness_score", "stress_score"]]
      .mean()
      .reset_index()
      .melt(id_vars="country", var_name="Metric", value_name="Average")
)
plt.figure(figsize=(12, 6))
sns.barplot(data=country_avg, x="Metric", y="Average", hue="country")
plt.title("Average Lifestyle & Mental Health Profile by Country")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# gender profile
gender_avg = (
    df.groupby("gender")[vars_all + ["happiness_score", "stress_score"]]
      .mean()
      .reset_index()
      .melt(id_vars="gender", var_name="Metric", value_name="Average")
)
plt.figure(figsize=(10, 6))
sns.barplot(data=gender_avg, x="Metric", y="Average", hue="gender")
plt.title("Average Lifestyle & Mental Health Profile by Gender")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# === 6. TOP vs. BOTTOM HAPPINESS COMPARISON ===
happiest = df.nlargest(10, "happiness_score")
saddest  = df.nsmallest(10, "happiness_score")
comparison = pd.DataFrame({
    "Happiest Avg": happiest[vars_all + ["stress_score"]].mean(),
    "Saddest Avg":  saddest[vars_all + ["stress_score"]].mean()
})
plt.figure(figsize=(8, 5))
comparison.plot(kind="bar")
plt.title("Average Lifestyle: Top 10 Happiest vs. 10 Least Happy")
plt.ylabel("Average Value")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

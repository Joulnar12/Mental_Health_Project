#Mental Health & Lifestyle Dataset - DATA 1
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# === 1. LOAD & CLEAN DATA ===
df = pd.read_csv("Mental_Health_Lifestyle_Dataset.csv")
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace(r'[()/%]', '', regex=True)

# === 2. DATA OVERVIEW ===
print("\n--- Dataset Shape ---")
print(df.shape)

print("\n--- Columns ---")
print(df.columns.tolist())

print("\n--- Missing Values ---")
print(df.isnull().sum())

# === 3. TRANSFORMATIONS ===
df['stress_level'] = df['stress_level'].str.strip().str.title()
stress_map = {'Low': 1, 'Moderate': 2, 'High': 3}
df['stress_score'] = df['stress_level'].map(stress_map)

df['happiness_score'] = pd.to_numeric(df['happiness_score'], errors='coerce')

# Encode exercise_level
exercise_map = {'Low': 1, 'Moderate': 2, 'High': 3}
df['exercise_score'] = df['exercise_level'].map(exercise_map)

# Binary encode presence of mental health condition
df['has_condition'] = df['mental_health_condition'].notnull().astype(int)

# === 4A. CORRELATION: HAPPINESS ===
happiness_vars = ['sleep_hours', 'screen_time_per_day_hours', 'work_hours_per_week',
                  'social_interaction_score', 'exercise_score', 'has_condition', 'happiness_score']
df_happy = df[happiness_vars].dropna()
corr_happy = df_happy.corr()

print("\n--- Correlation with Happiness Score ---")
print(corr_happy['happiness_score'].drop('happiness_score').sort_values(ascending=False))

plt.figure(figsize=(6, 4))
sns.heatmap(corr_happy[['happiness_score']], annot=True, cmap="Greens", fmt=".2f")
plt.title("Correlation with Happiness Score")
plt.tight_layout()
plt.show()

# === 4B. CORRELATION: STRESS ===
stress_vars = ['sleep_hours', 'screen_time_per_day_hours', 'work_hours_per_week',
               'social_interaction_score', 'exercise_score', 'has_condition', 'stress_score']
df_stress = df[stress_vars].dropna()
corr_stress = df_stress.corr()

print("\n--- Correlation with Stress Score ---")
print(corr_stress['stress_score'].drop('stress_score').sort_values(ascending=False))

plt.figure(figsize=(6, 4))
sns.heatmap(corr_stress[['stress_score']], annot=True, cmap="Reds", fmt=".2f")
plt.title("Correlation with Stress Score")
plt.tight_layout()
plt.show()

# === 5. CATEGORICAL PLOTS ===
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='exercise_level', hue='stress_level')
plt.title("Stress Level by Exercise Level")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
sns.boxplot(data=df, x='diet_type', y='happiness_score')
plt.title("Happiness Score by Diet Type")
plt.tight_layout()
plt.show()


print("\n--- Top 10 Happiest Individuals ---")
happiest = df.sort_values(by="happiness_score", ascending=False).head(10)
print(happiest[['country', 'gender', 'age', 'exercise_level', 'diet_type',
                'sleep_hours', 'stress_level', 'happiness_score']])

print("🔹 Avg profile (Top 10 Happy):")
print(happiest[['sleep_hours', 'work_hours_per_week', 'stress_score']].mean())

print("\n--- Bottom 10 Least Happy Individuals ---")
saddest = df.sort_values(by="happiness_score", ascending=True).head(10)
print(saddest[['country', 'gender', 'age', 'exercise_level', 'diet_type',
               'sleep_hours', 'stress_level', 'happiness_score']])
print("🔸 Avg profile (Bottom 10 Sad):")
print(saddest[['sleep_hours', 'work_hours_per_week', 'stress_score']].mean())


# === 7. PROFILE COMPARISON: HAPPIEST vs SADDEST ===

# Top and Bottom 10 based on happiness_score
happiest = df.sort_values(by="happiness_score", ascending=False).head(10)
saddest = df.sort_values(by="happiness_score", ascending=True).head(10)

# Select relevant numeric lifestyle features
features = ['sleep_hours', 'work_hours_per_week', 'screen_time_per_day_hours',
            'social_interaction_score', 'stress_score']

# Compute averages
happy_avg = happiest[features].mean()
sad_avg = saddest[features].mean()

# Combine for plotting
comparison_df = pd.DataFrame({'Happiest Avg': happy_avg, 'Saddest Avg': sad_avg})

# Plotting the comparison
plt.figure(figsize=(8, 5))
comparison_df.plot(kind='bar', figsize=(10, 6), colormap='Set2')
plt.title("Average Lifestyle Comparison: Top 10 Happiest vs Saddest")
plt.ylabel("Average Value")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# === Gender Profile Chart ===

# Prepare data: group by gender
gender_avg = df.groupby("gender")[['sleep_hours', 'work_hours_per_week',
                                   'screen_time_per_day_hours', 'social_interaction_score',
                                   'happiness_score', 'stress_score']].mean().reset_index()

# Melt into long format for grouped barplot
gender_melted = gender_avg.melt(id_vars='gender', var_name='Metric', value_name='Average Value')

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(data=gender_melted, x='Metric', y='Average Value', hue='gender')
plt.title("Average Lifestyle & Mental Health Profile by Gender")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# === Country Profile Chart ===

# Prepare data: group by country
country_avg = df.groupby("country")[['sleep_hours', 'work_hours_per_week',
                                     'screen_time_per_day_hours', 'social_interaction_score',
                                     'happiness_score', 'stress_score']].mean().reset_index()

# Melt for grouped barplot
country_melted = country_avg.melt(id_vars='country', var_name='Metric', value_name='Average Value')

# Plot
plt.figure(figsize=(12, 6))
sns.barplot(data=country_melted, x='Metric', y='Average Value', hue='country')
plt.title("Average Lifestyle & Mental Health Profile by Country")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

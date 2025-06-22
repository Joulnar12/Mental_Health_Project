#City Lifestyle Dataset - DATA 2
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Loading dataset
df = pd.read_csv("healthy_lifestyle_city_2021.csv")
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace(r'[()/%]', '', regex=True)

# === Dataset Overview ===
print("\n--- Dataset Shape ---")
print(df.shape)

print("\n--- Column Names ---")
print(df.columns)

print("\n--- Data Types ---")
print(df.dtypes)

print("\n--- Sample Data ---")
print(df.head())

print("\n--- Missing Values ---")
print(df.isnull().sum())

print("\n--- Unique Values (for object columns) ---")
for col in df.select_dtypes(include='object').columns:
    print(f"{col}: {df[col].unique()[:5]}...")  

# Step 1: Cleaning numeric columns (removed %, £, and convert to float)
df['obesity_rate'] = pd.to_numeric(df['obesity_levelscountry'].str.replace('%', ''), errors='coerce')
df['pollution_index'] = pd.to_numeric(df['pollutionindex_score_city'], errors='coerce')
df['gym_cost'] = pd.to_numeric(df['cost_of_a_monthly_gym_membershipcity'].str.replace('£', ''), errors='coerce')
df['bottle_water_cost'] = pd.to_numeric(df['cost_of_a_bottle_of_watercity'].str.replace('£', ''), errors='coerce')
df['work_hours'] = pd.to_numeric(df['annual_avg._hours_worked'], errors='coerce')
df['sunshine_hours'] = pd.to_numeric(df['sunshine_hourscity'], errors='coerce')
df['life_expectancy'] = pd.to_numeric(df['life_expectancyyears_country'], errors='coerce')
df['happiness'] = pd.to_numeric(df['happiness_levelscountry'], errors='coerce')

# Droping missing rows for correlation
df_clean = df.dropna(subset=['obesity_rate', 'pollution_index', 'gym_cost', 'work_hours', 'sunshine_hours', 'life_expectancy', 'happiness'])

# Step 2: Correlation analysis
correlation_features = ['obesity_rate', 'pollution_index', 'gym_cost', 'bottle_water_cost',
                        'work_hours', 'sunshine_hours', 'life_expectancy', 'happiness']
correlation_matrix = df[correlation_features].corr()

# Step 3: Correlation heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix - Lifestyle Factors and Happiness")
plt.tight_layout()
plt.show()

# Step 4: Scatter plots for key comparisons
sns.scatterplot(data=df, x='obesity_rate', y='happiness')
plt.title("Obesity Rate vs. Happiness")
plt.tight_layout()
plt.show()

sns.scatterplot(data=df, x='pollution_index', y='happiness')
plt.title("Pollution Index vs. Happiness")
plt.tight_layout()
plt.show()

sns.scatterplot(data=df, x='gym_cost', y='obesity_rate')
plt.title("Gym Cost vs. Obesity Rate")
plt.tight_layout()
plt.show()

sns.scatterplot(data=df, x='outdoor_activitiescity', y='happiness')
plt.title("Outdoor Activities vs. Happiness")
plt.tight_layout()
plt.show()

# Optional: Print top cities by happiness and obesity
print("\nTop 5 Happiest Cities:")
print(df[['city', 'happiness']].sort_values(by='happiness', ascending=False).head())

print("\nTop 5 Most Obese Countries (via Cities):")
print(df[['city', 'obesity_rate']].sort_values(by='obesity_rate', ascending=False).head())

print("\n--- Bottom 5 Happiest Cities ---")
print(df_clean[['city', 'happiness']].sort_values(by='happiness').head())

correlation_vars = ['gym_cost', 'bottle_water_cost', 'work_hours',
                    'pollution_index', 'sunshine_hours', 'obesity_rate', 'happiness']

# Removing missing values
df_corr = df[correlation_vars].dropna()

# Computing correlation matrix
corr_matrix = df_corr.corr()

# Print only correlations with happiness
print("\n--- Correlation of Economic & Environmental Factors with Happiness ---")
print(corr_matrix['happiness'].drop('happiness').sort_values(ascending=False))

plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Economic & Environmental Factors vs. Happiness")
plt.tight_layout()
plt.show()

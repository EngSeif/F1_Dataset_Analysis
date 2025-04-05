import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load your dataset
df = pd.read_excel("./F1_DataSet_Updated_Winners.xlsx")

# Ensure seaborn style
sns.set(style="whitegrid")

# 1. Race Position Summary by Pit Stop Timing (Descriptive Statistics)
print("\n--- Race Position Summary by Pit Stop Timing ---")
race_position_summary = df.groupby('Pit_Stop_Timing')['Race_Position'].describe()[['mean', '50%', 'std', 'count']]
race_position_summary.rename(columns={'50%': 'median'}, inplace=True)
print(race_position_summary)

# 1.1 Race Position Summary by Number of Pit Stops (Descriptive Statistics)
print("\n--- Race Position Summary by Number of Pit Stops ---")
race_position_by_pitstops = df.groupby('Number_of_Pit_Stops')['Race_Position'].describe()[['mean', '50%', 'std', 'count']]
race_position_by_pitstops.rename(columns={'50%': 'median'}, inplace=True)
print(race_position_by_pitstops)


# 2. Box Plot: Race Position by Pit Stop Timing (Already done)
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Pit_Stop_Timing', y='Race_Position', palette="Set2", hue='Pit_Stop_Timing', legend=False)
plt.title("Box Plot: Race Position by Pit Stop Timing")
plt.ylabel("Race Position")
plt.xlabel("Pit Stop Timing")
plt.show()

# 4. Histogram + KDE per Strategy (Normal Distributions)
plt.figure(figsize=(10, 6))
for strategy in df['Pit_Stop_Timing'].unique():
    sns.kdeplot(df[df['Pit_Stop_Timing'] == strategy]['Race_Position'], label=strategy, fill=True, linewidth=2)
plt.title("Race Position Distributions by Pit Stop Strategy (KDE)")
plt.xlabel("Race Position")
plt.legend()
plt.show()

# 5. Swarm Plot: Race Position by Pit Stop Timing
plt.figure(figsize=(10, 6))
sns.swarmplot(data=df, x='Pit_Stop_Timing', y='Race_Position', palette="Set2", hue='Pit_Stop_Timing', legend=False, size=2)
plt.title("Swarm Plot: Race Position by Pit Stop Timing")
plt.ylabel("Race Position")
plt.xlabel("Pit Stop Timing")
plt.show()

# 6. Facet Grid of Histograms for Race Positions by Pit Stop Strategy
g = sns.FacetGrid(df, col="Pit_Stop_Timing", height=4, col_wrap=2)
g.map(plt.hist, "Race_Position", bins=15, color='skyblue', edgecolor='black')
g.fig.suptitle("Race Position Histograms by Pit Stop Strategy", fontsize=16, y=1.05)
plt.show()

# 7. ANOVA: Testing if Pit Stop Timing affects Race Position
print("\n--- ANOVA Test ---")
anova_result = stats.f_oneway(
    df[df['Pit_Stop_Timing'] == 'Early']['Race_Position'],
    df[df['Pit_Stop_Timing'] == 'Late']['Race_Position'],
    df[df['Pit_Stop_Timing'] == 'Mid']['Race_Position'],
    df[df['Pit_Stop_Timing'] == 'Mixed']['Race_Position']
)
print(f"F-statistic: {anova_result.statistic}")
print(f"p-value: {anova_result.pvalue}")

# 8. Linear Regression: Modeling the relationship between Pit Stop Timing and Race Position
print("\n--- Linear Regression ---")

# Create dummy variables for Pit_Stop_Timing
df = pd.get_dummies(df, columns=['Pit_Stop_Timing'], drop_first=True)

# Build the regression model
model = smf.ols('Race_Position ~ Pit_Stop_Timing_Late + Pit_Stop_Timing_Mid + Pit_Stop_Timing_Mixed + Qualifying_Position + Number_of_Pit_Stops', data=df).fit()

# Print model summary
print(model.summary())

# 9. Check for normality of residuals
residuals = model.resid
sns.histplot(residuals, kde=True)
plt.title("Histogram of Residuals")
plt.show()


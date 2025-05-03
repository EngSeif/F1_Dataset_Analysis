import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

# -------------------------------
# 1. Data Loading & Cleaning
# -------------------------------
df = pd.read_excel('./merged_f1_data.xlsx', sheet_name='Sheet1')

# Convert pit stop duration to seconds
def convert_to_seconds(duration):
    try:
        if pd.isnull(duration):
            return np.nan
        if isinstance(duration, (int, float)):
            return float(duration)
        mins, secs = duration.split(':')
        if '.' in secs:
            secs, ms = secs.split('.')
            total_seconds = int(mins)*60 + int(secs) + int(ms)/1000
        else:
            total_seconds = int(mins)*60 + int(secs)
        return total_seconds
    except Exception:
        return np.nan

if 'duration' in df.columns:
    df['duration'] = df['duration'].apply(convert_to_seconds)

# -------------------------------
# 2. Feature Engineering
# -------------------------------
# Aggregate pit stop data per driver per race
pit_agg = df.groupby(['raceId', 'driverId']).agg(
    total_stops=('stop', 'max'),
    avg_pit_duration=('duration', 'mean'),
    first_pit_lap=('lap', 'min'),
    last_pit_lap=('lap', 'max')
).reset_index()

# Merge with final position data
final_pos = df[['raceId', 'driverId', 'position', 'qualifyingPos', 'constructorName']].drop_duplicates()
analysis_df = pd.merge(pit_agg, final_pos, on=['raceId', 'driverId'])

# Calculate race progress metrics
analysis_df['pit_window'] = analysis_df['last_pit_lap'] - analysis_df['first_pit_lap']
analysis_df['qualifying_gap'] = analysis_df['position'] - analysis_df['qualifyingPos']

# -------------------------------
# 3. Core Visualization
# -------------------------------
plt.figure(figsize=(12, 6))
sns.boxplot(x='total_stops', y='position', data=analysis_df)
plt.title('Final Position Distribution by Number of Pit Stops')
plt.xlabel('Total Pit Stops')
plt.ylabel('Final Race Position')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x='first_pit_lap', y='position', hue='total_stops',
                data=analysis_df, palette='viridis')
plt.title('First Pit Stop Lap vs Final Position')
plt.xlabel('First Pit Stop Lap')
plt.ylabel('Final Position')
plt.show()

# -------------------------------
# 4. Statistical Analysis
# -------------------------------
# Regression model 1: Position ~ Stops + Qualifying + Pit Duration
model1 = smf.ols('position ~ total_stops + qualifyingPos + avg_pit_duration',
                data=analysis_df).fit()
print("Regression Model 1: Position ~ Stops + Qualifying + Pit Duration\n")
print(model1.summary())

# Regression model 2: Including constructor effect
if 'constructorName' in analysis_df.columns:
    model2 = smf.ols('position ~ total_stops + qualifyingPos + C(constructorName)',
                    data=analysis_df).fit()
    print("\nRegression Model 2: Including Constructor Effect\n")
    print(model2.summary())

# T-test: Early vs late first pit stops
median_lap = analysis_df['first_pit_lap'].median()
early_stops = analysis_df[analysis_df['first_pit_lap'] <= median_lap]['position']
late_stops = analysis_df[analysis_df['first_pit_lap'] > median_lap]['position']
t_stat, p_value = ttest_ind(early_stops, late_stops, nan_policy='omit')
print(f"\nT-test: Early vs Late First Stops: t={t_stat:.2f}, p={p_value:.4f}")

sns.lmplot(x='qualifyingPos', y='position', data=analysis_df)
plt.title('Position vs Qualifying Position with Regression Line')
plt.show()

if 'constructorName' in analysis_df.columns:
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='constructorName', y='position', data=analysis_df)
    plt.title('Position Distribution by Constructor')
    plt.xticks(rotation=45)
    plt.show()

analysis_df['pit_group'] = analysis_df['first_pit_lap'].apply(lambda x: 'Early' if x <= median_lap else 'Late')

sns.boxplot(x='pit_group', y='position', data=analysis_df)
plt.title('Position by Early vs Late First Pit Stop')
plt.show()



# -------------------------------
# 5. Advanced Strategy Analysis
# -------------------------------
def analyze_pit_window(df):
    results = []
    for constructor in df['constructorName'].unique():
        const_df = df[df['constructorName'] == constructor]
        for stops in const_df['total_stops'].unique():
            stop_df = const_df[const_df['total_stops'] == stops]
            avg_position = stop_df['position'].mean()
            results.append({
                'Constructor': constructor,
                'Stops': stops,
                'AvgPosition': avg_position
            })
    return pd.DataFrame(results)

strategy_matrix = analyze_pit_window(analysis_df)
pivot_table = strategy_matrix.pivot(index='Constructor', columns='Stops', values='AvgPosition')

plt.figure(figsize=(12, 8))
sns.heatmap(pivot_table, annot=True, cmap='YlGnBu')
plt.title('Average Position by Constructor and Pit Stop Strategy')
plt.xlabel('Number of Pit Stops')
plt.ylabel('Constructor')
plt.show()




print("\n--- ANOVA Test: Impact of Pit Stop Count on Final Position ---")

# Aggregate data at driver-race level to avoid duplicate entries
aggregated_df = df.groupby(['raceId', 'driverId']).agg(
    total_stops=('stop', 'count'),
    avg_duration=('duration', 'mean'),
    qualifyingPos=('qualifyingPos', 'first'),
    position=('position', 'first')
).reset_index()

# Drop rows with missing values in relevant columns
anova_df = aggregated_df.dropna(subset=['total_stops', 'position'])

# Convert total_stops to categorical for ANOVA
# Only include common pit stop strategies (1-7 stops)
anova_df = anova_df[anova_df['total_stops'] <= 7]
anova_df['total_stops_cat'] = anova_df['total_stops'].astype('category')

# Perform one-way ANOVA: position ~ total_stops_cat
model = smf.ols('position ~ total_stops_cat', data=anova_df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print("\nANOVA Results (position ~ total_stops_cat):")
print(anova_table)

# Calculate mean positions for each pit stop count
means = anova_df.groupby('total_stops')['position'].mean().reset_index()
print("\nMean Final Positions by Number of Pit Stops:")
print(means)

# Visualize ANOVA results
plt.figure(figsize=(10, 6))
ax = sns.barplot(x='total_stops', y='position', data=anova_df,
                 estimator=np.mean, errorbar=('ci', 95))
plt.axhline(y=anova_df['position'].mean(), color='r', linestyle='--',
            label='Overall Mean')
plt.title('Mean Final Position by Number of Pit Stops with 95% CI')
plt.xlabel('Number of Pit Stops')
plt.ylabel('Mean Final Position')
plt.legend()
plt.show()

# Check if p-value is significant (threshold 0.05)
p_value = anova_table.loc['total_stops_cat', 'PR(>F)']
if p_value < 0.05:
    print(f"\nStatistically significant effect found (p={p_value:.4f})")
    print("Conclusion: Different pit stop counts lead to significantly different final positions")
else:
    print(f"\nNo statistically significant effect found (p={p_value:.4f})")
    print("Conclusion: Different pit stop counts do not significantly affect final positions")


# Aggregate data at driver-race level
# This prevents duplicate entries by combining all pit stops for each driver in each race
aggregated_df = df.groupby(['raceId', 'driverId']).agg(
    total_stops=('stop', 'count'),
    avg_duration=('duration', 'mean'),
    qualifyingPos=('qualifyingPos', 'first'),
    position=('position', 'first')
).reset_index()

# Remove rows with missing values
reg_df = aggregated_df.dropna(subset=['total_stops', 'avg_duration', 'qualifyingPos', 'position'])

# Run regression model: position ~ total_stops + qualifyingPos + avg_duration
model = smf.ols('position ~ total_stops + qualifyingPos + avg_duration', data=reg_df).fit()
print("\nRegression: position ~ total_stops + qualifyingPos + avg_duration\n")
print(model.summary())

# Create actual vs predicted plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x=model.predict(), y=reg_df['position'], alpha=0.7)
plt.plot([reg_df['position'].min(), reg_df['position'].max()],
         [reg_df['position'].min(), reg_df['position'].max()], 'r--')
plt.title('Actual vs Predicted Final Positions')
plt.xlabel('Predicted Position')
plt.ylabel('Actual Position')
plt.show()

# Create residuals plot
plt.figure(figsize=(10, 6))
sns.residplot(x=model.predict(), y=model.resid, lowess=True)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residuals vs Predicted Values')
plt.xlabel('Predicted Position')
plt.ylabel('Residuals')
plt.show()

# Interpretation
print("\nInterpretation of Regression Results:")
print("1. Total Stops Coefficient: Shows how each additional pit stop affects final position")
print("2. QualifyingPos Coefficient: Shows how starting position relates to final position")
print("3. Avg Duration Coefficient: Shows how pit stop efficiency impacts final position")
print("4. R-squared: Indicates what percentage of variation in final position is explained by the model")

# -------------------------------
# 6. Final Output
# -------------------------------
print("\n--- Analysis Complete ---")

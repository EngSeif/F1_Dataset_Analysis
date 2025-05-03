import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D

# Generate a dataset with strong correlation between ERS and lap times
def generate_f1_ers_dataset(n_samples=200):
    np.random.seed(42)  # For reproducibility
    
    # Generate ERS efficiency values (0.6 to 1.0 range)
    ers_efficiency = np.random.uniform(0.6, 1.0, n_samples)
    
    # Generate lap times with strong negative correlation to ERS
    lap_time_noise = np.random.normal(0, 1.5, n_samples)
    avg_lap_time = 90 - 10 * ers_efficiency + lap_time_noise
    
    # Generate race positions (correlated with lap time)
    race_positions = np.argsort(avg_lap_time) + 1
    race_positions = np.clip(race_positions + np.random.randint(-2, 3, n_samples), 1, 20)
    
    # Generate other relevant variables
    horsepower = 950 + 120 * ers_efficiency + np.random.normal(0, 15, n_samples)
    teams = ['Mercedes', 'Ferrari', 'Red Bull', 'McLaren', 'Alpine', 
             'Aston Martin', 'AlphaTauri', 'Alfa Romeo', 'Haas', 'Williams']
    engines = ['Mercedes', 'Ferrari', 'Honda', 'Renault']
    track_types = ['High-Speed', 'Technical', 'Mixed']
    
    # Create DataFrame
    df = pd.DataFrame({
        'Team': np.random.choice(teams, n_samples),
        'EngineManufacturer': np.random.choice(engines, n_samples),
        'TrackType': np.random.choice(track_types, n_samples),
        'Horsepower': horsepower.round(1),
        'ERS_Efficiency': ers_efficiency.round(4),
        'AvgLapTime': avg_lap_time.round(3),
        'RacePosition': race_positions,
        'EnergyRecoveryPerLap': (1.5 + 0.8 * ers_efficiency + np.random.normal(0, 0.1, n_samples)).round(3),
        'TopSpeed': (310 + 25 * ers_efficiency + np.random.normal(0, 5, n_samples)).round(1)
    })
    
    return df

# Generate the dataset
df = generate_f1_ers_dataset(150)

# Save to CSV for future use
df.to_csv('f1_ers_analysis_dataset.csv', index=False)

# Basic correlation analysis
print("=== Correlation Analysis ===")
ers_laptime_corr = df['ERS_Efficiency'].corr(df['AvgLapTime'])
ers_position_corr = df['ERS_Efficiency'].corr(df['RacePosition'])
print(f"ERS Efficiency vs Lap Time: r = {ers_laptime_corr:.3f}")
print(f"ERS Efficiency vs Race Position: r = {ers_position_corr:.3f}")

# Statistical tests
lap_corr, lap_p = stats.pearsonr(df['ERS_Efficiency'], df['AvgLapTime'])
pos_corr, pos_p = stats.pearsonr(df['ERS_Efficiency'], df['RacePosition'])
print(f"\nStatistical Tests:")
print(f"Lap Time Correlation: r = {lap_corr:.3f}, p = {lap_p:.6f}")
print(f"Race Position Correlation: r = {pos_corr:.3f}, p = {pos_p:.6f}")

# Regression analysis
X = sm.add_constant(df['ERS_Efficiency'])
lap_model = sm.OLS(df['AvgLapTime'], X).fit()

print("\n=== Regression Analysis: ERS Efficiency vs Lap Time ===")
print(lap_model.summary().tables[1])
print(f"Each 0.1 increase in ERS efficiency reduces lap time by {-lap_model.params[1]*0.1:.3f} seconds")

# VISUALIZATIONS

# 1. Correlation Heatmap
plt.figure(figsize=(10, 8))
corr_vars = ['Horsepower', 'ERS_Efficiency', 'AvgLapTime', 'RacePosition', 'EnergyRecoveryPerLap']
corr_matrix = df[corr_vars].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', center=0)
plt.title('Correlation Matrix: ERS Performance Metrics')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300)

# 2. ERS Efficiency vs Lap Time by Engine Manufacturer
plt.figure(figsize=(10, 6))
sns.scatterplot(x='ERS_Efficiency', y='AvgLapTime', hue='EngineManufacturer', 
                data=df, s=80, alpha=0.7)
plt.title('ERS Efficiency vs Average Lap Time by Engine Manufacturer')
plt.xlabel('ERS Efficiency')
plt.ylabel('Average Lap Time (seconds)')
plt.savefig('ers_vs_laptime_scatter.png', dpi=300)

# 3. Regression Plot with Confidence Interval
plt.figure(figsize=(10, 6))
sns.regplot(x='ERS_Efficiency', y='AvgLapTime', data=df, 
            scatter_kws={'alpha':0.6}, line_kws={'color':'red'})
plt.title(f'Regression: ERS Efficiency vs Lap Time (r = {lap_corr:.3f}, p < 0.001)')
plt.xlabel('ERS Efficiency')
plt.ylabel('Average Lap Time (seconds)')
plt.savefig('ers_laptime_regression.png', dpi=300)

# 4. 3D Visualization: Horsepower, ERS, Lap Time
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(df['Horsepower'], df['ERS_Efficiency'], df['AvgLapTime'],
                c=df['AvgLapTime'], cmap='viridis', s=50, alpha=0.7)
ax.set_xlabel('Horsepower')
ax.set_ylabel('ERS Efficiency')
ax.set_zlabel('Average Lap Time (s)')
plt.colorbar(sc, label='Lap Time (s)')
plt.title('3D Relationship: Horsepower, ERS Efficiency and Lap Time')
plt.savefig('3d_performance_plot.png', dpi=300)

# 5. Box Plots by Track Type
plt.figure(figsize=(12, 6))
sns.boxplot(x='TrackType', y='AvgLapTime', hue='EngineManufacturer', data=df)
plt.title('Lap Times by Track Type and Engine Manufacturer')
plt.xlabel('Track Type')
plt.ylabel('Average Lap Time (seconds)')
plt.savefig('laptime_by_track_type.png', dpi=300)

# 6. ERS Efficiency Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['ERS_Efficiency'], kde=True, bins=15)
plt.axvline(df['ERS_Efficiency'].mean(), color='r', linestyle='--', 
           label=f'Mean: {df["ERS_Efficiency"].mean():.3f}')
plt.title('Distribution of ERS Efficiency Values')
plt.xlabel('ERS Efficiency')
plt.ylabel('Frequency')
plt.legend()
plt.savefig('ers_efficiency_distribution.png', dpi=300)

print("\nAnalysis complete! Dataset and visualizations have been generated.")
print("The dataset has been saved as 'f1_ers_analysis_dataset.csv'")
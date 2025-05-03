import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy import stats

# Set seed for reproducibility
np.random.seed(42)


def generate_f1_data_extended(n_races=600):
    """Generate synthetic F1 race data with realistic relationships."""

    teams = ['Mercedes', 'Red Bull', 'Ferrari', 'McLaren', 'Alpine',
             'Aston Martin', 'AlphaTauri', 'Alfa Romeo', 'Williams', 'Haas']
    team_performance = {
        'Mercedes': -0.3, 'Red Bull': -0.25, 'Ferrari': -0.2, 'McLaren': -0.1,
        'Alpine': 0.0, 'Aston Martin': 0.05, 'AlphaTauri': 0.1,
        'Alfa Romeo': 0.15, 'Williams': 0.2, 'Haas': 0.25
    }
    track_types = ['High-Speed', 'Mixed', 'Technical']
    weather_conditions = ['Dry', 'Mixed', 'Wet']
    data = []

    for _ in range(n_races):
        team = np.random.choice(teams)
        track_type = np.random.choice(track_types)
        weather = np.random.choice(weather_conditions, p=[0.7, 0.2, 0.1])
        base_perf = team_performance[team]

        # Downforce level influenced by team philosophy
        if team in ['Red Bull', 'Ferrari', 'Mercedes']:
            downforce = np.clip(0.78 + np.random.normal(0, 0.05), 0.6, 0.95)
        elif team in ['McLaren', 'Alpine', 'Aston Martin']:
            downforce = np.clip(0.75 + np.random.normal(0, 0.05), 0.6, 0.95)
        else:
            downforce = np.clip(0.72 + np.random.normal(0, 0.05), 0.6, 0.95)

        # Horsepower
        if team in ['Mercedes', 'Ferrari', 'Red Bull']:
            horsepower = np.clip(np.random.normal(1020, 20), 900, 1050)
        else:
            horsepower = np.clip(np.random.normal(980, 30), 900, 1050)

        # Calculate cornering speed (km/h) - positively correlated with downforce
        base_corner_speed = 180  # base cornering speed
        cornering_speed = base_corner_speed + 100 * (downforce - 0.7)  # main effect
        if track_type == 'Technical':
            cornering_speed += 10  # technical tracks favor cornering
        if weather == 'Wet':
            cornering_speed -= 15  # wet reduces cornering speed
        cornering_speed += np.random.normal(0, 3)  # noise

        # Calculate straight line speed (km/h) - negatively correlated with downforce
        base_straight_speed = 320
        straight_speed = base_straight_speed - 150 * (downforce - 0.7)  # main effect
        if track_type == 'High-Speed':
            straight_speed += 10
        if weather == 'Wet':
            straight_speed -= 10
        straight_speed += np.random.normal(0, 5)  # noise

        # Average lap time calculation
        base_lap_time = 88.0 + base_perf
        if track_type == 'High-Speed':
            downforce_effect = 3 * (downforce - 0.7) ** 2
        elif track_type == 'Technical':
            downforce_effect = -1.5 * (downforce - 0.7)
        else:
            downforce_effect = 2 * (downforce - 0.78) ** 2

        if weather == 'Wet':
            downforce_effect -= (downforce - 0.65) * 1.2
            base_lap_time += 1.5
        elif weather == 'Mixed':
            downforce_effect -= (downforce - 0.7) * 0.8
            base_lap_time += 0.8

        power_effect = -0.5 * np.log(horsepower / 950)
        lap_time = base_lap_time + downforce_effect + power_effect + np.random.normal(0, 0.3)
        lap_time = np.clip(lap_time, 86.0, 92.0)

        # Race position based on lap time with noise
        position_base = 1 + (lap_time - 86.0) * 2
        race_position = int(np.clip(round(position_base + np.random.normal(0, 2)), 1, 20))

        data.append({
            'Team': team,
            'Track_Type': track_type,
            'Weather': weather,
            'Downforce_Level': round(downforce, 3),
            'Horsepower_bhp': round(horsepower, 1),
            'Cornering_Speed': round(cornering_speed, 2),
            'Straight_Line_Speed': round(straight_speed, 2),
            'Average_Lap_Time': round(lap_time, 3),
            'Race_Position': race_position
        })

    df = pd.DataFrame(data)
    df['Downforce_Category'] = pd.qcut(df['Downforce_Level'], q=3, labels=['Low', 'Medium', 'High'])
    return df


# Function to calculate regression coefficients
def calc_regression_coefficients(df, variable):
    X = df['Downforce_Level'].values.reshape(-1, 1)
    y = df[variable].values
    model = LinearRegression().fit(X, y)
    r2 = model.score(X, y)
    coef = float(model.coef_[0])
    intercept = float(model.intercept_)
    return {
        'variable': variable,
        'coefficient': coef,
        'intercept': intercept,
        'r_squared': r2,
        'equation': f"{variable} = {intercept:.2f} + {coef:.2f} × Downforce_Level"
    }


# Generate data
f1_data = generate_f1_data_extended(600)

from scipy.stats import mode

def safe_mode(series):
    """Return the mode of a series, or NaN if all values are unique."""
    m = mode(series, nan_policy='omit', keepdims=False)
    return m.mode if m.count > 1 else np.nan

# Variables to analyze
variables = ['Downforce_Level', 'Straight_Line_Speed', 'Cornering_Speed', 'Race_Position']

# Calculate statistics
stats_summary = {}
for var in variables:
    data = f1_data[var]
    stats_summary[var] = {
        'Mean': data.mean(),
        'Median': data.median(),
        'Mode': safe_mode(data),
        'Variance': data.var(),
        'Std_Dev': data.std()
    }

# Convert to DataFrame for better visualization
stats_df = pd.DataFrame(stats_summary).T
print(stats_df)

# 1. Race Position Analysis
plt.figure(figsize=(12, 7))
sns.boxplot(data=f1_data, x='Downforce_Category', y='Race_Position', hue='Track_Type')
plt.gca().invert_yaxis()  # Invert y-axis so better positions (lower numbers) are at the top
plt.title('Race Position by Downforce Category and Track Type')
plt.xlabel('Downforce Level')
plt.ylabel('Race Position (lower is better)')
plt.savefig('position_by_downforce.png')
plt.show()

# 2. REGRESSION SCATTER PLOTS
print("REGRESSION ANALYSIS:")
print("-" * 50)

variables = ['Average_Lap_Time', 'Cornering_Speed', 'Straight_Line_Speed', 'Race_Position']
labels = ['Average Lap Time (seconds)', 'Cornering Speed (km/h)',
          'Straight Line Speed (km/h)', 'Race Position']

regression_results = {}
for var, label in zip(variables, labels):
    # Calculate regression
    X = f1_data['Downforce_Level'].values.reshape(-1, 1)
    y = f1_data[var].values
    model = LinearRegression().fit(X, y)

    # Generate prediction line
    x_range = np.linspace(f1_data['Downforce_Level'].min(), f1_data['Downforce_Level'].max(), 100).reshape(-1, 1)
    y_pred = model.predict(x_range)

    # Plot
    plt.figure(figsize=(12, 7))
    plt.scatter(f1_data['Downforce_Level'], f1_data[var], alpha=0.5, color='skyblue')
    plt.plot(x_range, y_pred, 'r-', linewidth=2, label='Linear Regression')
    plt.title(f'Downforce Level vs {label}', fontsize=14)
    plt.xlabel('Downforce Level', fontsize=12)
    plt.ylabel(label, fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

    # Store and print results
    regression_results[var] = {
        'coefficient': float(model.coef_[0]),
        'intercept': float(model.intercept_),
        'r_squared': model.score(X, y)
    }

    print(f"{var} vs Downforce Level:")
    print(f"  Coefficient: {model.coef_[0]:.4f}")
    print(f"  Intercept: {model.intercept_:.4f}")
    print(f"  R-squared: {model.score(X, y):.4f}")
    print(f"  Equation: {var} = {model.intercept_:.4f} + ({model.coef_[0]:.4f} × Downforce_Level)")
    print()

# 3. HEAT MAPS
print("\nHEAT MAP ANALYSIS:")
print("-" * 50)

for var, label in zip(variables, labels):
    pivot_data = f1_data.pivot_table(
        index='Downforce_Category',
        columns='Track_Type',
        values=var,
        aggfunc='mean'
    )

    plt.figure(figsize=(10, 6))
    heatmap = sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='coolwarm',
                          cbar_kws={'label': label})
    plt.title(f'Average {label} by Downforce Category and Track Type', fontsize=14)
    plt.ylabel('Downforce Category', fontsize=12)
    plt.xlabel('Track Type', fontsize=12)
    plt.show()

    print(f"Heatmap values for {label}:")
    print(pivot_data)
    print()
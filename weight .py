import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# Define constant values for the variables (removed HYBRID_VS_NON_HYBRID)
WEATHER_CONDITIONS = 'Mixed'
RACE_LENGTH = 305
ERS_DEPLOYMENT = 'Standard'
GEARBOX_TYPE = 'Standard'
TIRE_COMPOUND = 'Soft'
DOWNFORCE_LEVEL = 6
DRAG_COEFFICIENT = 0.7
MAJOR_REGULATION_CHANGES = False

# Load Excel data
df = pd.read_excel('f1_design_analysis_dataset_2018_2025 (1).xlsx')

# Extract and clean relevant data for regression analysis
useful_data = df[['Chassis_Weight_kg', 'Average_Lap_Time_sec']].dropna()

# Check if data is available for regression
if useful_data.empty:
    print("No data available for the selected races.")
else:
    # Compute variance
    variance_weight = useful_data['Chassis_Weight_kg'].var()
    variance_lap_time = useful_data['Average_Lap_Time_sec'].var()

    # Compute Pearson correlation and p-value
    pearson_corr, p_value = pearsonr(useful_data['Chassis_Weight_kg'], useful_data['Average_Lap_Time_sec'])

    # Perform linear regression
    model = LinearRegression()
    X = useful_data[['Chassis_Weight_kg']]
    y = useful_data['Average_Lap_Time_sec']
    model.fit(X, y)

    # Get regression parameters
    slope = model.coef_[0]
    intercept = model.intercept_
    r_squared = model.score(X, y)

    # Print results
    print(f"Pearson Correlation (Chassis Weight vs Lap Time): {pearson_corr:.3f}")
    print(f"P-value for Correlation: {p_value:.3f}")
    print(f"Slope of Regression Line: {slope:.3f} seconds/kg")
    print(f"Intercept: {intercept:.3f}")
    print(f"R-squared Value: {r_squared:.3f}")
    print(f"Variance of Chassis Weight (Weather {WEATHER_CONDITIONS}): {variance_weight:.3f}")
    print(f"Variance of Average Lap Time (Weather {WEATHER_CONDITIONS}): {variance_lap_time:.3f}")

    # Scatter plot with regression line
    plt.figure(figsize=(8, 5))
    plt.scatter(X, y, alpha=0.5)
    plt.plot(X, model.predict(X), color='red')
    plt.xlabel('Chassis Weight (kg)')
    plt.ylabel('Average Lap Time (s)')
    plt.title(f'Weight vs Lap Time Relationship (Weather {WEATHER_CONDITIONS})')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Correlation matrix heatmap
    plt.figure(figsize=(6, 4))
    corr_matrix = useful_data.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.show()

# Calculate and display overall lap time distribution per Year as a box plot
# Drop missing values in 'Year' or 'Average_Lap_Time_sec'
yearly_data = df[['Year', 'Average_Lap_Time_sec']].dropna()

# Print a summary of lap times per year if needed
print("\nLap Time Distribution per Year:")
print(yearly_data.groupby('Year')['Average_Lap_Time_sec'].describe())

# Create a box plot of Average Lap Time per Year
plt.figure(figsize=(10, 6))
sns.boxplot(data=yearly_data, x='Year', y='Average_Lap_Time_sec')
plt.xlabel('Year')
plt.ylabel('Average Lap Time (s)')
plt.title('Lap Time Distribution per Year')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

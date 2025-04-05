# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy.stats import pearsonr

# # Load data from Excel
# file_path = "f1_design_analysis_with_ERS_2018_2025 (1).xlsx"  # Update with your file path
# df = pd.read_excel(file_path)

# # Clean column names to remove any spaces or special characters
# df.columns = df.columns.str.strip()

# # Define the variables
# ers_performance = df["ERS_Deployment_Efficiency"]  # Independent variable
# lap_times = df["Average_Lap_Time_sec"]  # Dependent variable

# # Calculate Pearson correlation
# correlation, p_value = pearsonr(ers_performance, lap_times)

# # Print the Pearson correlation and p-value
# print(f"Pearson Correlation: {correlation:.3f}")
# print(f"P-value: {p_value:.3f}")

# # Interpret the correlation
# if correlation > 0:
#     print("There is a positive correlation: As ERS Efficiency increases, Lap Time increases.")
# elif correlation < 0:
#     print("There is a negative correlation: As ERS Efficiency increases, Lap Time decreases.")
# else:
#     print("There is no significant linear correlation between ERS Efficiency and Lap Time.")

# # Plot data with regression line
# plt.figure(figsize=(8, 6))
# sns.regplot(x=ers_performance, y=lap_times, scatter_kws={"color": "blue"}, line_kws={"color": "red"})
# plt.xlabel("ERS Efficiency (%)")
# plt.ylabel("Average Lap Time (seconds)")
# plt.title(f"ERS Efficiency vs. Average Lap Time (Correlation: {correlation:.3f})")
# plt.grid()
# plt.show()

import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

# Define the race (country) you want to analyze
CIRCUIT_TYPE = 'Hungarian GP'  # Example: assuming the Hungarian GP as the target race
START_YEAR = 2018
END_YEAR = 2024

# Load Excel data
df = pd.read_excel('f1_design_with_ers_analysis.xlsx')

# Filter data for the specified race and years (2018-2024)
df_race = df[(df['Race_Name'] == CIRCUIT_TYPE) & (df['Year'] >= START_YEAR) & (df['Year'] <= END_YEAR)]

# Extract and clean relevant data (ERS Efficiency and Average Lap Time)
useful_data = df_race[['ERS_Efficiency', 'Average_Lap_Time_sec']].dropna()

# Check if data is available
if useful_data.empty:
    print(f"No data available for the {CIRCUIT_TYPE} from {START_YEAR} to {END_YEAR}.")
else:
    # Compute variance
    variance_ers = useful_data['ERS_Efficiency'].var()
    variance_lap_time = useful_data['Average_Lap_Time_sec'].var()

    # Compute correlation
    correlation = useful_data.corr().iloc[0, 1]

    # Perform linear regression
    model = LinearRegression()
    X = useful_data[['ERS_Efficiency']]  # Independent variable (ERS Efficiency)
    y = useful_data['Average_Lap_Time_sec']  # Dependent variable (Average Lap Time)
    model.fit(X, y)

    # Results
    print(f"ERS_Efficiency - Lap Time Correlation ({CIRCUIT_TYPE}, {START_YEAR}-{END_YEAR}): {correlation:.3f}")
    print(f"Variance of ERS_Efficiency ({CIRCUIT_TYPE}, {START_YEAR}-{END_YEAR}): {variance_ers:.3f}")
    print(f"Variance of Average Lap Time ({CIRCUIT_TYPE}, {START_YEAR}-{END_YEAR}): {variance_lap_time:.3f}")
    print(f"Lap time improvement per unit ERS_Efficiency increase ({CIRCUIT_TYPE}, {START_YEAR}-{END_YEAR}): {abs(model.coef_[0]):.3f} seconds/% ERS Efficiency")
    print(f"R² value ({CIRCUIT_TYPE}, {START_YEAR}-{END_YEAR}): {model.score(X, y):.3f}")

    # Scatter Plot with Regression Line
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, alpha=0.5)
    plt.plot(X, model.predict(X), color='red', label="Trend Line")
    plt.xlabel('ERS_Efficiency (%)')
    plt.ylabel('Average Lap Time (s)')
    plt.title(f'ERS_Efficiency vs Lap Time Relationship ({CIRCUIT_TYPE}, {START_YEAR}-{END_YEAR})')
    plt.legend()
    plt.show()

    # Heatmap of the correlation matrix (without 'Year')
    correlation_matrix = useful_data.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title(f'Correlation Matrix: ERS Efficiency vs Lap Time ({CIRCUIT_TYPE}, {START_YEAR}-{END_YEAR})')
    plt.show()

    # Better Distribution Histogram for ERS Efficiency
    plt.figure(figsize=(10, 6))
    sns.histplot(df_race['ERS_Efficiency'], bins=30, kde=True, color='blue', alpha=0.7)
    plt.title(f'Distribution of ERS Efficiency (%) ({CIRCUIT_TYPE}, {START_YEAR}-{END_YEAR})')
    plt.xlabel('ERS Efficiency (%)')
    plt.ylabel('Frequency')
    plt.show()

    # Better Distribution Histogram for Average Lap Time
    plt.figure(figsize=(10, 6))
    sns.histplot(df_race['Average_Lap_Time_sec'], bins=30, kde=True, color='green', alpha=0.7)
    plt.title(f'Distribution of Average Lap Time (s) ({CIRCUIT_TYPE}, {START_YEAR}-{END_YEAR})')
    plt.xlabel('Average Lap Time (s)')
    plt.ylabel('Frequency')
    plt.show()

    # Box Plot for ERS Efficiency by Year
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Year', y='ERS_Efficiency', data=df_race, showfliers=False)
    plt.title(f'Box Plot of ERS Efficiency by Year ({CIRCUIT_TYPE}, {START_YEAR}-{END_YEAR})')
    plt.xlabel('Year')
    plt.ylabel('ERS Efficiency (%)')
    plt.show()

    # Box Plot for Average Lap Time by Year
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Year', y='Average_Lap_Time_sec', data=df_race, showfliers=False)
    plt.title(f'Box Plot of Average Lap Time by Year ({CIRCUIT_TYPE}, {START_YEAR}-{END_YEAR})')
    plt.xlabel('Year')
    plt.ylabel('Average Lap Time (s)')
    plt.show()

    # Trend Over Time: Average ERS Efficiency and Lap Time by Year
    plt.figure(figsize=(10, 6))
    avg_data = df_race.groupby('Year')[['ERS_Efficiency', 'Average_Lap_Time_sec']].mean().reset_index()
    plt.plot(avg_data['Year'], avg_data['ERS_Efficiency'], label='Average ERS Efficiency', marker='o')
    plt.plot(avg_data['Year'], avg_data['Average_Lap_Time_sec'], label='Average Lap Time', marker='o')
    plt.xlabel('Year')
    plt.ylabel('Value')
    plt.title(f'Trend Over Time ({CIRCUIT_TYPE}, {START_YEAR}-{END_YEAR})')
    plt.legend()
    plt.show()

    # Residual Plot (Fixing the issue with fitting residuals)
    plt.figure(figsize=(10, 6))
    residuals = y - model.predict(X)
    sns.residplot(x=model.predict(X), y=residuals, color='purple', line_kws={'color': 'red'})
    plt.title(f'Residual Plot: ERS Efficiency vs Lap Time ({CIRCUIT_TYPE}, {START_YEAR}-{END_YEAR})')
    plt.xlabel('Fitted Values (Predicted Lap Time)')
    plt.ylabel('Residuals (Actual - Predicted)')
    plt.show()

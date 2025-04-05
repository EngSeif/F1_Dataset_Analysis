# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.linear_model import LinearRegression
# import numpy as np

# # Load the Excel file
# file_path = r"C:\Users\alini\Downloads\F1_DataSet_2018_2022_Modified.xlsx"  # Update this with the actual file path
# df = pd.read_excel(file_path, sheet_name="Sheet1")

# # Extract required columns
# X = df[['Budget_Millions_USD']].values  # Independent variable (Budget)
# y = df['Race_Position'].values  # Dependent variable (Race Position)

# # Scatter Plot with Regression Line
# plt.figure(figsize=(8, 5))
# sns.regplot(x=X.flatten(), y=y, scatter_kws={"alpha": 0.5}, line_kws={"color": "red"})

# # Labels and title
# plt.xlabel("Team Budget (Millions USD)")
# plt.ylabel("Race Position (Lower is Better)")
# plt.title("Budget vs. Race Position in F1")

# # Show the plot
# plt.show()

# # Perform Simple Linear Regression
# model = LinearRegression()
# model.fit(X, y)

# # Print regression results
# print(f"Intercept: {model.intercept_}")
# print(f"Slope (Budget Coefficient): {model.coef_[0]}")
# print(f"R-squared: {model.score(X, y)}")


# import pandas as pd
# from sklearn.linear_model import LinearRegression
# import matplotlib.pyplot as plt

# # Define constant values for the variables
# CIRCUIT_TYPE = 'US GP'  # Example: assuming data is from Hungarian GP
# WEATHER_CONDITIONS = 'Mixed'  # For title/context purposes

# # Load Excel data from the modified dataset file
# df = pd.read_excel(r"C:\Users\alini\Downloads\F1_DataSet_2018_2022_Modified.xlsx")


# # Filter data for the specific race (if desired)
# df_race = df[df['Race_Name'] == CIRCUIT_TYPE]  # If you want to compare within a specific race

# # Extract and clean relevant data
# useful_data = df_race[['Budget_Millions_USD', 'Race_Position']].dropna()

# # Check if data is available
# if useful_data.empty:
#     print(f"No data available for the {CIRCUIT_TYPE}.")
# else:
#     # Compute variance for each variable
#     variance_budget = useful_data['Budget_Millions_USD'].var()
#     variance_position = useful_data['Race_Position'].var()

#     # Compute correlation between Budget and Race Position
#     correlation = useful_data.corr().iloc[0, 1]

#     # Perform linear regression analysis
#     model = LinearRegression()
#     X = useful_data[['Budget_Millions_USD']]
#     y = useful_data['Race_Position']
#     model.fit(X, y)

#     # Results
#     print(f"Budget-Race Position Correlation ({CIRCUIT_TYPE}, Weather {WEATHER_CONDITIONS}): {correlation:.3f}")
#     print(f"Variance of Budget (Millions USD) ({CIRCUIT_TYPE}, Weather {WEATHER_CONDITIONS}): {variance_budget:.3f}")
#     print(f"Variance of Race Position ({CIRCUIT_TYPE}, Weather {WEATHER_CONDITIONS}): {variance_position:.3f}")
#     print(f"Race Position change per Million USD budget increase ({CIRCUIT_TYPE}, Weather {WEATHER_CONDITIONS}): {model.coef_[0]:.3f} positions/Million USD")
#     print(f"R² value ({CIRCUIT_TYPE}, Weather {WEATHER_CONDITIONS}): {model.score(X, y):.3f}")

#     # Visualization of the relationship
#     plt.scatter(X, y, alpha=0.5)
#     plt.plot(X, model.predict(X), color='red')
#     plt.xlabel('Budget (Millions USD)')
#     plt.ylabel('Race Position')
#     plt.title(f'Budget vs Race Position Relationship ({CIRCUIT_TYPE}, Weather {WEATHER_CONDITIONS})')
#     plt.show()

# import pandas as pd
# from sklearn.linear_model import LinearRegression
# import matplotlib.pyplot as plt
# from scipy import stats

# # Define constant values for the variables
# CIRCUIT_TYPE = 'US GP'  # Example: data is from US GP
# WEATHER_CONDITIONS = 'Mixed'  # For title/context purposes

# # Load Excel data from the modified dataset file
# df = pd.read_excel(r"C:\Users\alini\Downloads\F1_DataSet_2018_2022_Modified.xlsx")

# # Filter data for the specific race
# df_race = df[df['Race_Name'] == CIRCUIT_TYPE]

# # Extract and clean relevant data
# useful_data = df_race[['Budget_Millions_USD', 'Race_Position']].dropna()

# if useful_data.empty:
#     print(f"No data available for the {CIRCUIT_TYPE}.")
# else:
#     # Compute the full correlation matrix for the useful data
#     correlation_matrix = useful_data.corr()
#     print("\nCorrelation Matrix:")
#     print(correlation_matrix)
    
#     # Calculate Pearson correlation and p-value for Budget vs Race Position
#     pearson_corr, p_val_corr = stats.pearsonr(useful_data['Budget_Millions_USD'], useful_data['Race_Position'])
    
#     # Perform linear regression analysis using scikit-learn
#     model = LinearRegression()
#     X = useful_data[['Budget_Millions_USD']]
#     y = useful_data['Race_Position']
#     model.fit(X, y)
    
#     # Extract regression parameters
#     slope = model.coef_[0]
#     intercept = model.intercept_
#     r_squared = model.score(X, y)
    
#     # Print required outputs
#     print("\nRegression and Correlation Metrics:")
#     print(f"Pearson Correlation: {pearson_corr:.3f}")
#     print(f"P-value for Correlation: {p_val_corr:.3f}")
#     print(f"Slope of Regression Line: {slope:.3f}")
#     print(f"Intercept: {intercept:.3f}")
#     print(f"R-squared Value: {r_squared:.3f}")
    
#     # Visualization
#     plt.figure(figsize=(12, 5))
    
#     # Subplot 1: Scatter plot with regression line
#     plt.subplot(1, 2, 1)
#     plt.scatter(X, y, alpha=0.5)
#     plt.plot(X, model.predict(X), color='red')
#     plt.xlabel('Budget (Millions USD)')
#     plt.ylabel('Race Position')
#     plt.title(f'Budget vs Race Position ({CIRCUIT_TYPE}, Weather {WEATHER_CONDITIONS})')
    
#     # Subplot 2: Heatmap of the correlation matrix with numerical annotations
#     plt.subplot(1, 2, 2)
#     heatmap = plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='none')
#     plt.colorbar(heatmap)
#     plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=45)
#     plt.yticks(range(len(correlation_matrix.index)), correlation_matrix.index)
#     # Annotate each cell with the numeric value
#     for i in range(len(correlation_matrix.index)):
#         for j in range(len(correlation_matrix.columns)):
#             plt.text(j, i, f"{correlation_matrix.iloc[i, j]:.2f}",
#                      ha="center", va="center", color="black")
#     plt.title('Correlation Matrix')
    
#     plt.tight_layout()
#     plt.show()


# import pandas as pd
# from sklearn.linear_model import LinearRegression
# import matplotlib.pyplot as plt
# from scipy import stats

# # Define constant values for the variables
# CIRCUIT_TYPE = 'US GP'  # Example: data is from US GP
# WEATHER_CONDITIONS = 'Mixed'  # For title/context purposes

# # Load Excel data from the modified dataset file
# df = pd.read_excel(r"C:\Users\alini\Downloads\F1_DataSet_2018_2022_Modified.xlsx")

# # Filter data for the specific race
# df_race = df[df['Race_Name'] == CIRCUIT_TYPE]

# # Extract and clean relevant data
# useful_data = df_race[['Budget_Millions_USD', 'Race_Position']].dropna()

# if useful_data.empty:
#     print(f"No data available for the {CIRCUIT_TYPE}.")
# else:
#     # Compute the full correlation matrix for the useful data
#     correlation_matrix = useful_data.corr()
#     print("\nCorrelation Matrix:")
#     print(correlation_matrix)
    
#     # Calculate Pearson correlation and p-value for Budget vs Race Position
#     pearson_corr, p_val_corr = stats.pearsonr(useful_data['Budget_Millions_USD'], useful_data['Race_Position'])
    
#     # Perform linear regression analysis using scikit-learn
#     model = LinearRegression()
#     X = useful_data[['Budget_Millions_USD']]
#     y = useful_data['Race_Position']
#     model.fit(X, y)
    
#     # Extract regression parameters
#     slope = model.coef_[0]
#     intercept = model.intercept_
#     r_squared = model.score(X, y)
    
#     # Print required outputs
#     print("\nRegression and Correlation Metrics:")
#     print(f"Pearson Correlation: {pearson_corr:.3f}")
#     print(f"P-value for Correlation: {p_val_corr:.3f}")
#     print(f"Slope of Regression Line: {slope:.3f}")
#     print(f"Intercept: {intercept:.3f}")
#     print(f"R-squared Value: {r_squared:.3f}")
    
#     # Visualization: Scatter plot with regression line and heatmap for the correlation matrix
#     plt.figure(figsize=(12, 5))
    
#     # Subplot 1: Scatter plot with regression line
#     plt.subplot(1, 2, 1)
#     plt.scatter(X, y, alpha=0.5)
#     plt.plot(X, model.predict(X), color='red')
#     plt.xlabel('Budget (Millions USD)')
#     plt.ylabel('Race Position')
#     plt.title(f'Budget vs Race Position ({CIRCUIT_TYPE}, Weather {WEATHER_CONDITIONS})')
    
#     # Subplot 2: Heatmap of the correlation matrix with numerical annotations
#     plt.subplot(1, 2, 2)
#     heatmap = plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='none')
#     plt.colorbar(heatmap)
#     plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=45)
#     plt.yticks(range(len(correlation_matrix.index)), correlation_matrix.index)
#     # Annotate each cell with the numeric value
#     for i in range(len(correlation_matrix.index)):
#         for j in range(len(correlation_matrix.columns)):
#             plt.text(j, i, f"{correlation_matrix.iloc[i, j]:.2f}",
#                      ha="center", va="center", color="black")
#     plt.title('Correlation Matrix')
    
#     plt.tight_layout()
#     plt.show()
    
#     # Additional Visualizations: Budget Distribution and Budget Distribution by Year
#     plt.figure(figsize=(12, 5))
    
#     # Subplot 1: Overall Budget Distribution Histogram
#     plt.subplot(1, 2, 1)
#     plt.hist(useful_data['Budget_Millions_USD'], bins=10, color='skyblue', edgecolor='black')
#     plt.xlabel('Budget (Millions USD)')
#     plt.ylabel('Frequency')
#     plt.title('Overall Budget Distribution')
    
#     # Subplot 2: Budget Distribution by Year (if 'Year' column exists)
#     if 'Year' in df_race.columns:
#         plt.subplot(1, 2, 2)
#         # Group data by year and prepare list of budget values for each year
#         years = sorted(df_race['Year'].dropna().unique())
#         budget_by_year = [df_race[df_race['Year'] == year]['Budget_Millions_USD'].dropna() for year in years]
#         plt.boxplot(budget_by_year, labels=years)
#         plt.xlabel('Year')
#         plt.ylabel('Budget (Millions USD)')
#         plt.title('Budget Distribution by Year')
#     else:
#         print("The 'Year' column is not available in the dataset to plot distribution by year.")
    
#     plt.tight_layout()
#     plt.show()


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from scipy import stats

# Define constant values for the variables
CIRCUIT_TYPE = 'US GP'  # Example: data is from US GP
WEATHER_CONDITIONS = 'Mixed'  # For title/context purposes

# Load Excel data from the modified dataset file
df = pd.read_excel(r"C:\Users\alini\Downloads\F1_DataSet_2018_2022_Modified.xlsx")

# Filter data for the specific race
df_race = df[df['Race_Name'] == CIRCUIT_TYPE]

# Extract and clean relevant data
useful_data = df_race[['Budget_Millions_USD', 'Race_Position']].dropna()

if useful_data.empty:
    print(f"No data available for the {CIRCUIT_TYPE}.")
else:
    # Compute the full correlation matrix for the useful data
    correlation_matrix = useful_data.corr()
    print("\nCorrelation Matrix:")
    print(correlation_matrix)
    
    # Calculate Pearson correlation and p-value for Budget vs Race Position
    pearson_corr, p_val_corr = stats.pearsonr(useful_data['Budget_Millions_USD'], useful_data['Race_Position'])
    
    # Perform linear regression analysis using scikit-learn
    model = LinearRegression()
    X = useful_data[['Budget_Millions_USD']]
    y = useful_data['Race_Position']
    model.fit(X, y)
    
    # Extract regression parameters
    slope = model.coef_[0]
    intercept = model.intercept_
    r_squared = model.score(X, y)
    
    # Print required outputs
    print("\nRegression and Correlation Metrics:")
    print(f"Pearson Correlation: {pearson_corr:.3f}")
    print(f"P-value for Correlation: {p_val_corr:.3f}")
    print(f"Slope of Regression Line: {slope:.3f}")
    print(f"Intercept: {intercept:.3f}")
    print(f"R-squared Value: {r_squared:.3f}")
    
    # Visualization: Scatter plot with regression line and heatmap for the correlation matrix
    plt.figure(figsize=(12, 5))
    
    # Subplot 1: Scatter plot with regression line
    plt.subplot(1, 2, 1)
    plt.scatter(X, y, alpha=0.5)
    plt.plot(X, model.predict(X), color='red')
    plt.xlabel('Budget (Millions USD)')
    plt.ylabel('Race Position')
    plt.title(f'Budget vs Race Position ({CIRCUIT_TYPE}, Weather {WEATHER_CONDITIONS})')
    
    # Subplot 2: Heatmap of the correlation matrix with numerical annotations
    plt.subplot(1, 2, 2)
    heatmap = plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='none')
    plt.colorbar(heatmap)
    plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=45)
    plt.yticks(range(len(correlation_matrix.index)), correlation_matrix.index)
    # Annotate each cell with the numeric value
    for i in range(len(correlation_matrix.index)):
        for j in range(len(correlation_matrix.columns)):
            plt.text(j, i, f"{correlation_matrix.iloc[i, j]:.2f}",
                     ha="center", va="center", color="black")
    plt.title('Correlation Matrix')
    
    plt.tight_layout()
    plt.show()
    
    # Additional Visualizations: Budget Distribution and Budget Distribution by Year
    plt.figure(figsize=(12, 5))
    
    # Subplot 1: Overall Budget Distribution Histogram with Density Curve Guidance
    plt.subplot(1, 2, 1)
    # Plot histogram with density normalization
    x_values = useful_data['Budget_Millions_USD']
    plt.hist(x_values, bins=10, density=True, color='skyblue', edgecolor='black')
    # Calculate and plot the KDE curve
    density = stats.gaussian_kde(x_values)
    xs = np.linspace(x_values.min(), x_values.max(), 200)
    plt.plot(xs, density(xs), color='red', lw=2, label='Density Curve')
    plt.xlabel('Budget (Millions USD)')
    plt.ylabel('Density')
    plt.title('Overall Budget Distribution')
    plt.legend()
    
    # Subplot 2: Budget Distribution by Year (if 'Year' column exists)
    if 'Year' in df_race.columns:
        plt.subplot(1, 2, 2)
        # Group data by year and prepare list of budget values for each year
        years = sorted(df_race['Year'].dropna().unique())
        budget_by_year = [df_race[df_race['Year'] == year]['Budget_Millions_USD'].dropna() for year in years]
        plt.boxplot(budget_by_year, labels=years)
        plt.xlabel('Year')
        plt.ylabel('Budget (Millions USD)')
        plt.title('Budget Distribution by Year')
    else:
        print("The 'Year' column is not available in the dataset to plot distribution by year.")
    
    plt.tight_layout()
    plt.show()





import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import pearsonr, ttest_ind
from sklearn.preprocessing import PolynomialFeatures

# Load dataset
df = pd.read_excel('f1_weight_lap_dataset.xlsx')

# 1. Basic Data Exploration
print("Dataset Overview:")
print(df.head())
print("\nDescriptive Statistics:")
print(df[['CarWeight_kg', 'FuelCorrectedLapTime_s']].describe())

# 2. Correlation Analysis
corr, p_value = pearsonr(df['CarWeight_kg'], df['FuelCorrectedLapTime_s'])
print(f"\nPearson Correlation: {corr:.3f} (p-value: {p_value:.4f})")

# 3. Visualization
plt.figure(figsize=(12, 8))
# plt.figure(figsize=(12, 8))

# Scatter plot with regression line
plt.subplot(2, 2, 1)
sns.regplot(x='CarWeight_kg', y='FuelCorrectedLapTime_s', data=df, 
            scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
plt.title('Weight vs Lap Time Relationship')
plt.xlabel('Car Weight (kg)')
plt.ylabel('Fuel-Corrected Lap Time (s)')

# Distribution plots
plt.subplot(2, 2, 2)
sns.histplot(df['CarWeight_kg'], kde=True, color='blue')
plt.title('Car Weight Distribution')

plt.subplot(2, 2, 3)
sns.histplot(df['FuelCorrectedLapTime_s'], kde=True, color='green')
plt.title('Lap Time Distribution')

plt.tight_layout()
plt.show()

# 4. Regression Analysis
# Simple Linear Regression
X = df['CarWeight_kg']
y = df['FuelCorrectedLapTime_s']
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print("\nSimple Linear Regression Results:")
print(model.summary())

# Multiple Regression with Control Variables
X_multi = df[['CarWeight_kg', 'EnginePower_HP', 'Downforce_CL']]
X_multi = sm.add_constant(X_multi)
model_multi = sm.OLS(y, X_multi).fit()
print("\nMultiple Regression Results:")
print(model_multi.summary())

# 5. Advanced Analysis
# Weight categories analysis
df['Weight_Category'] = pd.cut(df['CarWeight_kg'], 
                              bins=[780, 790, 800, 810, 820],
                              labels=['780-790', '790-800', '800-810', '810-820'])

plt.figure(figsize=(10, 6))
sns.boxplot(x='Weight_Category', y='FuelCorrectedLapTime_s', data=df)
plt.title('Lap Time Distribution by Weight Categories')
plt.xlabel('Weight Categories (kg)')
plt.ylabel('Lap Time (s)')
plt.show()

# T-test between lightest and heaviest categories
light = df[df['Weight_Category'] == '780-790']['FuelCorrectedLapTime_s']
heavy = df[df['Weight_Category'] == '810-820']['FuelCorrectedLapTime_s']
t_stat, p_val = ttest_ind(light, heavy)
print(f"\nT-test between lightest and heaviest categories: t-stat = {t_stat:.2f}, p-value = {p_val:.4f}")

# 6. Non-linear Relationship Check
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(df[['CarWeight_kg']])
model_poly = sm.OLS(y, X_poly).fit()
print("\nPolynomial Regression Results:")
print(model_poly.summary())

# 7. Partial Regression Plot
fig, ax = plt.subplots(figsize=(8, 6))  # Create figure and axes
sm.graphics.plot_partregress(
    endog='FuelCorrectedLapTime_s',
    exog_i='CarWeight_kg',
    exog_others=['EnginePower_HP', 'Downforce_CL'],
    data=df,
    obs_labels=False,
    ax=ax  # Pass the axes object here
)
plt.title('Partial Regression Plot (Controlling for Other Variables)')
plt.show()

# 8. Residual Analysis
plt.figure(figsize=(10, 6))
plt.scatter(model_multi.fittedvalues, model_multi.resid, alpha=0.5)
plt.axhline(y=0, color='red', linestyle='--')
plt.title('Residuals vs Fitted Values')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.show()


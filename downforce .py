# Formula 1 Design Impact Analysis
# Question: How do variations in downforce levels affect race performance in Formula 1?

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy import stats

# ======================================
# 1. Load Dataset
# ======================================
df = pd.read_excel('F1_DataSet_2018_2022.xlsx')

# Clean and preprocess
df = df.dropna(subset=["Downforce_Level", "Average_Lap_Time_sec", "Race_Position"])
df["Race_Position"] = pd.to_numeric(df["Race_Position"], errors="coerce")

# ======================================
# 2. Descriptive Statistics
# ======================================
print("Descriptive Statistics:\n")
print(df[["Downforce_Level", "Average_Lap_Time_sec", "Race_Position"]].describe())

# ======================================
# 3. Correlation Matrix
# ======================================
print("\nCorrelation Matrix:\n")
corr_matrix = df[["Downforce_Level", "Average_Lap_Time_sec", "Race_Position"]].corr()
print(corr_matrix)

# ======================================
# 4. Visualizations
# ======================================
sns.set(style="whitegrid")

# Histogram of Downforce
plt.figure(figsize=(6, 4))
sns.histplot(df["Downforce_Level"], kde=True, color="steelblue")
plt.title("Distribution of Downforce Levels")
plt.xlabel("Downforce Level")
plt.tight_layout()
plt.show()

# Boxplot: Downforce vs Lap Time
plt.figure(figsize=(6, 4))
sns.boxplot(x=pd.qcut(df["Downforce_Level"], 4), y="Average_Lap_Time_sec", data=df)
plt.title("Average Lap Time by Downforce Quartiles")
plt.xlabel("Downforce Quartile")
plt.ylabel("Average Lap Time (sec)")
plt.tight_layout()
plt.show()

# Scatter + Regression Line: Downforce vs Lap Time
plt.figure(figsize=(6, 4))
sns.regplot(x="Downforce_Level", y="Average_Lap_Time_sec", data=df, scatter_kws={'s': 40}, line_kws={'color': 'red'})
plt.title("Downforce Level vs Average Lap Time (with Regression Line)")
plt.xlabel("Downforce Level")
plt.ylabel("Average Lap Time (sec)")
plt.tight_layout()
plt.show()

# Scatter + Regression Line: Downforce vs Race Position
plt.figure(figsize=(6, 4))
sns.regplot(x="Downforce_Level", y="Race_Position", data=df, scatter_kws={'s': 40}, line_kws={'color': 'red'})
plt.title("Downforce Level vs Race Position (with Regression Line)")
plt.xlabel("Downforce Level")
plt.ylabel("Race Position (Lower is Better)")
plt.tight_layout()
plt.show()

# Correlation Heatmap
plt.figure(figsize=(5, 4))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# ======================================
# 5. Linear Regression & Hypothesis Testing
# ======================================

# Regression 1: Downforce -> Average Lap Time
X1 = sm.add_constant(df["Downforce_Level"])
y1 = df["Average_Lap_Time_sec"]
model1 = sm.OLS(y1, X1).fit()
print("\nLinear Regression: Downforce vs Lap Time")
print(model1.summary())

# Regression 2: Downforce -> Race Position
X2 = sm.add_constant(df["Downforce_Level"])
y2 = df["Race_Position"]
model2 = sm.OLS(y2, X2).fit()
print("\nLinear Regression: Downforce vs Race Position")
print(model2.summary())
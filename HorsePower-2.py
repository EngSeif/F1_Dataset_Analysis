import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import pingouin as pg
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import boxcox
from sklearn.linear_model import LinearRegression

# 1. Load and prepare data
df = pd.read_excel('Enhanced_F1_Horsepower_Analysis_2019-2025.xlsx')

# Data cleaning and transformations
df = df.dropna(subset=['Horsepower (HP)', 'Avg Qualifying Position', 'Season Budget (M$)'])
numeric_cols = ['Horsepower (HP)', 'Avg Qualifying Position', 'Season Budget (M$)']
df[numeric_cols] = df[numeric_cols].astype(float)
df['HP_transformed'], _ = boxcox(df['Horsepower (HP)'] + 1)
df['HP_Budget_Interaction'] = df['Horsepower (HP)'] * (df['Season Budget (M$)'] / 100)

# 2. Enhanced Descriptive Statistics
print("=== Enhanced Descriptive Statistics ===")
print(df[numeric_cols].describe())
print("\n=== Engine Manufacturer Distribution ===")
print(df['Engine Manufacturer'].value_counts())

# 3. Comprehensive Correlation Analysis with Heatmap
plt.figure(figsize=(10,8))
corr_matrix = df[numeric_cols + ['HP_Budget_Interaction']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title("Correlation Heatmap of Key Variables")
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300)
plt.show()

# 4. Distribution Plots
fig, ax = plt.subplots(1, 2, figsize=(12,5))
sns.histplot(df['Horsepower (HP)'], kde=True, ax=ax[0])
sns.histplot(df['Avg Qualifying Position'], kde=True, ax=ax[1])
ax[0].set_title("Horsepower Distribution")
ax[1].set_title("Qualifying Position Distribution")
plt.tight_layout()
plt.savefig('distribution_plots.png', dpi=300)
plt.show()

# 5. Advanced Regression Analysis with CI
X = sm.add_constant(df['Horsepower (HP)'])
y = df['Avg Qualifying Position']
model = sm.OLS(y, X).fit()
print("\n=== Regression Results with Confidence Intervals ===")
print(model.summary())
print("\n95% Confidence Intervals for Coefficients:")
print(model.conf_int(alpha=0.05))

# 6. ANOVA for Engine Manufacturers
print("\n=== ANOVA: Engine Manufacturer Impact ===")
anova_results = pg.anova(data=df, dv='Avg Qualifying Position', between='Engine Manufacturer')
print(anova_results)

# 7. Enhanced Visualizations
# Pairplot for multivariate relationships
sns.pairplot(df[numeric_cols + ['Engine Manufacturer']], 
             hue='Engine Manufacturer', palette='viridis')
plt.suptitle("Multivariate Relationship Analysis", y=1.02)
plt.savefig('pairplot_analysis.png', dpi=300)
plt.show()

# 8. Residual Analysis
fig, ax = plt.subplots(1, 2, figsize=(12,5))
sns.residplot(x=model.predict(), y=model.resid, lowess=True, ax=ax[0])
ax[0].set_title("Residuals vs Fitted Values")
sm.qqplot(model.resid, line='s', ax=ax[1])
ax[1].set_title("Q-Q Plot of Residuals")
plt.tight_layout()
plt.savefig('residual_analysis.png', dpi=300)
plt.show()

# 9. Machine Learning Interpretation
rf = RandomForestRegressor(n_estimators=200)
X_rf = df[['Horsepower (HP)', 'Season Budget (M$)', 'HP_Budget_Interaction']]
y_rf = df['Avg Qualifying Position']
rf.fit(X_rf, y_rf)

plt.figure(figsize=(8,5))
sns.barplot(x=rf.feature_importances_, y=X_rf.columns, palette='viridis')
plt.title("Random Forest Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.savefig('feature_importance.png', dpi=300)
plt.show()

# 10. Hypothesis Testing Suite
print("\n=== Comprehensive Hypothesis Testing ===")
# Pearson correlation test
corr_coef, p_val = stats.pearsonr(df['Horsepower (HP)'], df['Avg Qualifying Position'])
print(f"Pearson Correlation: r = {corr_coef:.3f}, p = {p_val:.4f}")

# T-test for regression coefficient
t_stat = model.tvalues[1]
p_value = model.pvalues[1]
print(f"\nRegression Coefficient t-test: t = {t_stat:.2f}, p = {p_value:.4f}")

# F-test for model significance
f_value = model.fvalue
f_pvalue = model.f_pvalue
print(f"Model F-test: F = {f_value:.2f}, p = {f_pvalue:.4f}")

# Bootstrapped confidence intervals
def bootstrap_corr(data, n_bootstrap=1000):
    corrs = []
    for _ in range(n_bootstrap):
        sample = data.sample(frac=1, replace=True)
        corr = sample['Horsepower (HP)'].corr(sample['Avg Qualifying Position'])
        corrs.append(corr)
    return np.percentile(corrs, [2.5, 97.5])

ci = bootstrap_corr(df)
print(f"\nBootstrapped 95% CI for Pearson r: [{ci[0]:.3f}, {ci[1]:.3f}]")

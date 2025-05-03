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

# Enhanced data cleaning
df = df.dropna(subset=['Horsepower (HP)', 'Avg Qualifying Position', 'Season Budget (M$)'])
df['Avg Qualifying Position'] = df['Avg Qualifying Position'].astype(float)
df['Horsepower (HP)'] = df['Horsepower (HP)'].astype(float)
df['Season Budget (M$)'] = df['Season Budget (M$)'].astype(float)

# Data transformations
df['HP_transformed'], _ = boxcox(df['Horsepower (HP)'] + 1)  # Normalize horsepower
df['HP_Budget_Interaction'] = df['Horsepower (HP)'] * (df['Season Budget (M$)'] / 100)

# 2. Advanced descriptive statistics
print("=== Enhanced Descriptive Statistics ===")
print(df[['Horsepower (HP)', 'Avg Qualifying Position', 'Season Budget (M$)']].describe())

# 3. Comprehensive correlation analysis
print("\n=== Multivariate Correlation Analysis ===")
corr_matrix = df[['Horsepower (HP)', 'Avg Qualifying Position', 
                'Season Budget (M$)', 'HP_Budget_Interaction']].corr(method='spearman')
print(corr_matrix)

# 4. Polynomial regression with interaction terms
print("\n=== Polynomial Regression Results ===")
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(df[['Horsepower (HP)', 'Season Budget (M$)']])
model_poly = sm.OLS(df['Avg Qualifying Position'], sm.add_constant(X_poly)).fit()
print(model_poly.summary())

# 5. Mixed effects model accounting for team and engine
print("\n=== Mixed Effects Model (Team Random Effects) ===")
model_mixed = smf.mixedlm("Q('Avg Qualifying Position') ~ Q('Horsepower (HP)') + Q('Season Budget (M$)')",
                         data=df, 
                         groups=df['Team']).fit()
print(model_mixed.summary())

# 6. Machine learning approach
print("\n=== Random Forest Feature Importance ===")
rf = RandomForestRegressor(n_estimators=200)
X_rf = df[['Horsepower (HP)', 'Season Budget (M$)', 'HP_Budget_Interaction']]
y_rf = df['Avg Qualifying Position']
rf.fit(X_rf, y_rf)
print(f"R-squared: {rf.score(X_rf, y_rf):.3f}")
print("Feature importances:", dict(zip(X_rf.columns, rf.feature_importances_)))

# 7. Enhanced visualizations
plt.figure(figsize=(14, 6))

# First calculate partial correlation
partial_corr = pg.partial_corr(data=df, x='Horsepower (HP)', 
                              y='Avg Qualifying Position', 
                              covar='Season Budget (M$)')

# Then create partial regression plot
def partial_residual(x, y, covar):
    model_x = LinearRegression().fit(covar.values.reshape(-1,1), x)
    res_x = x - model_x.predict(covar.values.reshape(-1,1))
    
    model_y = LinearRegression().fit(covar.values.reshape(-1,1), y)
    res_y = y - model_y.predict(covar.values.reshape(-1,1))
    
    return res_x, res_y

res_hp, res_qual = partial_residual(df['Horsepower (HP)'], 
                                   df['Avg Qualifying Position'],
                                   df['Season Budget (M$)'])

plt.subplot(1, 2, 1)
sns.regplot(x=res_hp, y=res_qual, scatter_kws={'alpha':0.6}, line_kws={'color':'red'})
plt.title(f"Partial Regression Plot (Controlling for Budget)\n"
          f"r = {partial_corr['r'].values[0]:.2f}, p = {partial_corr['p-val'].values[0]:.4f}")
plt.xlabel("Horsepower (HP | Budget)")
plt.ylabel("Qualifying Position | Budget")

# Rest of the code remains the same...


# 3D visualization
ax = plt.subplot(1, 2, 2, projection='3d')
ax.scatter(df['Horsepower (HP)'], df['Season Budget (M$)'], df['Avg Qualifying Position'],
           c=pd.factorize(df['Engine Manufacturer'])[0], cmap='viridis')
ax.set_xlabel('Horsepower (HP)')
ax.set_ylabel('Budget (M$)')
ax.set_zlabel('Qualifying Position')
plt.title("3D Relationship Analysis")
plt.tight_layout()
plt.savefig('advanced_analysis.png', dpi=300)
plt.show()

# 8. Engine-specific analysis
g = sns.FacetGrid(df, col="Engine Manufacturer", col_wrap=2, height=4)
g.map_dataframe(sns.regplot, x='Horsepower (HP)', y='Avg Qualifying Position', 
                scatter_kws={'alpha':0.6}, line_kws={'color':'red'})
g.set_titles("{col_name} Engine")
plt.savefig('engine_specific.png', dpi=300)
plt.show()

# 9. Time series analysis with confidence intervals
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x='Season', y='Horsepower (HP)', 
             errorbar=('ci', 95), label='Horsepower')
sns.lineplot(data=df, x='Season', y='Avg Qualifying Position', 
             errorbar=('ci', 95), label='Qualifying Position')
plt.title("Temporal Trends with 95% Confidence Intervals")
plt.xlabel("Season")
plt.ylabel("Value")
plt.legend()
plt.savefig('time_series_ci.png', dpi=300)
plt.show()

# 10. Advanced hypothesis testing
print("\n=== Bootstrapped Correlation Analysis ===")
def bootstrap_corr(data, n_bootstrap=1000):
    corrs = []
    for _ in range(n_bootstrap):
        sample = data.sample(frac=1, replace=True)
        corr = sample['Horsepower (HP)'].corr(sample['Avg Qualifying Position'])
        corrs.append(corr)
    return np.percentile(corrs, [2.5, 97.5])

ci = bootstrap_corr(df)
print(f"95% CI for Pearson r: [{ci[0]:.3f}, {ci[1]:.3f}]")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

from statsmodels.stats.diagnostic import het_breuschpagan

# =====================================================
# 1. LOAD DATA
# =====================================================

file_path = "/storage/emulated/0/Download/linear.csv"

df = pd.read_csv(file_path)

print("\n========== FIRST 5 ROWS ==========")
print(df.head())

# =====================================================
# 2. DATA CLEANING
# =====================================================

print("\n========== DATA INFO ==========")
print(df.info())

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

# Remove missing values
df = df.dropna()

print("\n========== DATA AFTER CLEANING ==========")
print(df.shape)

# =====================================================
# 3. DESCRIPTIVE STATISTICS
# =====================================================

print("\n========== DESCRIPTIVE STATISTICS ==========")
print(df.describe())

# =====================================================
# 4. CORRELATION MATRIX
# =====================================================

print("\n========== CORRELATION MATRIX ==========")
corr_matrix = df.corr()
print(corr_matrix)

plt.figure(figsize=(8,6))
sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Correlation Matrix")
plt.show()

# =====================================================
# 5. SCATTERPLOTS (LINEARITY CHECK)
# =====================================================

plt.figure(figsize=(6,4))
plt.scatter(df['TV'], df['Sales'])
plt.xlabel('TV')
plt.ylabel('Sales')
plt.title('TV vs Sales')
plt.show()

plt.figure(figsize=(6,4))
plt.scatter(df['Radio'], df['Sales'])
plt.xlabel('Radio')
plt.ylabel('Sales')
plt.title('Radio vs Sales')
plt.show()

plt.figure(figsize=(6,4))
plt.scatter(df['Social_Media'], df['Sales'])
plt.xlabel('Social_Media')
plt.ylabel('Sales')
plt.title('Social Media vs Sales')
plt.show()

# Optional Pairplot
sns.pairplot(df)
plt.show()

# =====================================================
# 6. MODEL BUILDING
# =====================================================

X = df[['TV', 'Radio', 'Social_Media']]
y = df['Sales']

X = sm.add_constant(X)

ols_model = sm.OLS(y, X).fit()

print("\n========== OLS SUMMARY ==========")
print(ols_model.summary())

# =====================================================
# 7. COEFFICIENTS & CONFIDENCE INTERVALS
# =====================================================

print("\n========== MODEL COEFFICIENTS ==========")
print(ols_model.params)

print("\n========== CONFIDENCE INTERVALS ==========")
print(ols_model.conf_int())

# =====================================================
# 8. FINAL REGRESSION EQUATION
# =====================================================

intercept = ols_model.params['const']
tv_coef = ols_model.params['TV']
radio_coef = ols_model.params['Radio']
social_coef = ols_model.params['Social_Media']

print("\n========== FINAL REGRESSION EQUATION ==========")

print(
    f"Sales = {intercept:.4f} "
    f"+ ({tv_coef:.4f} × TV) "
    f"+ ({radio_coef:.4f} × Radio) "
    f"+ ({social_coef:.4f} × Social_Media)"
)

# =====================================================
# 9. PREDICTIONS & RESIDUALS
# =====================================================

predicted = ols_model.fittedvalues
residuals = ols_model.resid

# =====================================================
# 10. RESIDUALS VS FITTED
# HOMOSCEDASTICITY CHECK
# =====================================================

plt.figure(figsize=(8,5))

plt.scatter(predicted, residuals)

plt.axhline(
    y=0,
    color='red',
    linestyle='--'
)

plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs Fitted Values")
plt.show()

# =====================================================
# 11. HISTOGRAM OF RESIDUALS
# NORMALITY CHECK
# =====================================================

plt.figure(figsize=(8,5))

plt.hist(
    residuals,
    bins=20
)

plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.title("Histogram of Residuals")
plt.show()

# =====================================================
# 12. Q-Q PLOT
# NORMALITY CHECK
# =====================================================

sm.qqplot(
    residuals,
    line='45'
)

plt.title("Q-Q Plot of Residuals")
plt.show()

# =====================================================
# 13. BREUSCH-PAGAN TEST
# HOMOSCEDASTICITY TEST
# =====================================================

bp_test = het_breuschpagan(
    residuals,
    X
)

bp_labels = [
    "LM Statistic",
    "LM-Test p-value",
    "F-Statistic",
    "F-Test p-value"
]

print("\n========== BREUSCH-PAGAN TEST ==========")

for label, value in zip(bp_labels, bp_test):
    print(f"{label}: {value}")

# =====================================================
# 14. MODEL PERFORMANCE
# =====================================================

print("\n========== MODEL PERFORMANCE ==========")

print(f"R-squared: {ols_model.rsquared:.4f}")
print(f"Adjusted R-squared: {ols_model.rsquared_adj:.4f}")

# =====================================================
# 15. BUSINESS INTERPRETATION
# =====================================================

print("\n========== BUSINESS INTERPRETATION ==========")

print(
    "The coefficients indicate the expected increase "
    "in Sales for a one-unit increase in each marketing "
    "channel while holding other variables constant."
)

print(
    "The channel with the largest statistically "
    "significant coefficient has the strongest impact "
    "on Sales."
)

print(
    "R-squared explains the percentage of variation "
    "in Sales explained by TV, Radio and Social Media."
)
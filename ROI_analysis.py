import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
#
# 1. LOAD DATA
#

file_path = "/storage/emulated/0/Download/linear.csv"

df = pd.read_csv(file_path)

print(df.head())

#
# 2. DATA CLEANING
#
print(df.info())
print(df.isnull().sum())

# Remove missing values
df = df.dropna()
print(df.shape)
#
# 3. DESCRIPTIVE STATISTICS
#
print(df.describe())

#
#4. SCATTERPLOTS (LINEARITY CHECK)
#
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

#
# 5. MODEL BUILDING
#
X = df[['TV', 'Radio', 'Social_Media']]
y = df['Sales']
print(y)


# Fitting OLS Model
X = sm.add_constant(X)
ols_model = sm.OLS(y,X). fit()
print (ols_model.summary())
#
#COEFFICIENT AND CONFIDENT INTERVAL
#

print(ols_model.params)
print(ols_model.conf_int())

#
# 6. PREDICTION AND RESIDUAL
#
residuals = ols_model.resid
predicted = ols_model.fittedvalues

#
# 7. RESIDUALS VS FITTED
# HOMOSCEDASTICITY CHECK

plt.figure(figsize=(8,5))
plt.scatter(predicted, residuals)

plt.axhline(y=0,
            color='red',
            linestyle='--')

plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted Values')

plt.show()

# 8. HISTOGRAM OF RESIDUALS
# NORMALITY CHECK
# 


plt.figure(figsize=(8,5))

plt.hist(
    residuals,
    bins=20
)

plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.title("Histogram of Residuals")
plt.show()


# 9. Q-Q PLOT
# NORMALITY CHECK
# 

sm.qqplot(
    residuals,
    line='45'
)

plt.title("Q-Q Plot of Residuals")
plt.show()
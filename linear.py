import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import statsmodels.api as sm

file_path = "/storage/emulated/0/Download/linear.csv"

df = pd.read_csv(file_path)

df = df.dropna()
print(df.head())

X = df[['TV', 'Radio', 'Social_Media']]
y = df['Sales']
print(y)


# Fitting OLS Model
X = sm.add_constant(X)
ols_model = sm.OLS(y,X). fit()
print (ols_model.summary())

residuals = ols_model.resid
predicted = ols_model.fittedvalues


plt.figure(figsize=(8,5))
plt.scatter(predicted, residuals)

plt.axhline(y=0,
            color='red',
            linestyle='--')

plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted Values')

plt.show()
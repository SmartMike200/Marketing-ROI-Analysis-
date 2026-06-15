# Marketing-ROI-Analysis-
Analyzed a marketing dataset using Python and statsmodels to build a Simple Linear Regression model.
## Executive Summary
We analyzed the impact of three marketing channels:
TV Advertising
Radio Advertising
Social Media Advertising
on Sales.
The model explains 99.9% of the variation in sales (R² = 0.999), which means the model predicts sales extremely well.
Key Finding
TV advertising is the only channel with a statistically significant impact on sales.
Radio and Social Media showed little to no measurable impact on sales in this dataset.

## Channel-by-Channel Interpretation
TV Advertising
Coefficient = 3.5626
p-value = 0.000
**Business interpretation:**
For every additional ₦1 spent on TV advertising, sales increase by approximately ₦3.56, assuming all other factors remain constant.
This result is highly significant and reliable.
Radio Advertising
Coefficient = -0.0040
p-value = 0.685
**Business interpretation:**
Radio advertising showed no meaningful impact on sales. The observed effect is so small that it could easily be due to random variation.
Social Media Advertising
Coefficient = 0.0050
p-value = 0.842
**Business interpretation:**
Social media advertising also showed no statistically significant impact on sales in this analysis.

## ROI Recommendation
Strongest ROI Impact: TV Advertising
Among all channels:
Channel
Impact on Sales
Significant?
TV = +3.5626

Radio = -0.0040

Social Media = +0.0050

### Recommendation
1. If the goal is to maximize sales:
Increase investment in TV advertising.

2. Review Radio campaigns before allocating additional budget.

3. Audit Social Media strategy, targeting, and content because current spending is not translating into measurable sales.

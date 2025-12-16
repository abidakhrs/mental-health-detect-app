import pandas as pd
import numpy as np

# Load data
df = pd.read_csv("Impact_of_Remote_Work_on_Mental_Health.csv")

# Create Binary Target: Burnout (1) vs Others (0)
# Note: Ensure 'Mental_Health_Condition' uses the same logic as the app
df['Target'] = df['Mental_Health_Condition'].apply(lambda x: 1 if x == 'Burnout' else 0)

# 1. Correlation between Age and Burnout
correlation = df['Age'].corr(df['Target'])
print(f"Correlation between Age and Burnout Risk: {correlation:.4f}")

# 2. Burnout Rate by Age Group
# Bin ages into groups: <25, 25-34, 35-44, 45-54, 55+
bins = [0, 25, 35, 45, 55, 100]
labels = ['<25', '25-34', '35-44', '45-54', '55+']
df['Age_Group'] = pd.cut(df['Age'], bins=bins, labels=labels)

print("\nBurnout Rate by Age Group:")
age_group_stats = df.groupby('Age_Group', observed=False)['Target'].mean()
print(age_group_stats)

print("\nCount by Age Group:")
print(df['Age_Group'].value_counts(sort=False))

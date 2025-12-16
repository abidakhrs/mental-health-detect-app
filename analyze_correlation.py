import pandas as pd

df = pd.read_csv("Impact_of_Remote_Work_on_Mental_Health.csv")
print("Crosstab Stress vs Mental Health:")
print(pd.crosstab(df['Stress_Level'], df['Mental_Health_Condition'], normalize='index'))

print("\nCrosstab Sleep vs Mental Health:")
print(pd.crosstab(df['Sleep_Quality'], df['Mental_Health_Condition'], normalize='index'))

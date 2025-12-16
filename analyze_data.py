import pandas as pd
import numpy as np

try:
    df = pd.read_csv("Impact_of_Remote_Work_on_Mental_Health.csv")
    print("Columns:", df.columns.tolist())
    print("\nTarget Distribution:")
    if 'Mental_Health_Condition' in df.columns:
        print(df['Mental_Health_Condition'].value_counts(normalize=True))
    else:
        print("Target column 'Mental_Health_Condition' not found!")
        
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    print("\nUnique Values:")
    for col in ['Work_Location', 'Stress_Level', 'Access_to_Mental_Health_Resources', 
                'Productivity_Change', 'Satisfaction_with_Remote_Work', 
                'Physical_Activity', 'Sleep_Quality']:
        if col in df.columns:
            print(f"{col}: {df[col].unique()}")
    
except Exception as e:
    print(e)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load data
df = pd.read_csv("Impact_of_Remote_Work_on_Mental_Health.csv")

# 1. Handle Missing Values
df = df.dropna(subset=['Mental_Health_Condition'])
df['Physical_Activity'] = df['Physical_Activity'].fillna('No')

# 2. Encode Features
sleep_quality_map = {'Poor': 1, 'Average': 2, 'Good': 3}
stress_level_map = {'Low': 1, 'Medium': 2, 'High': 3}
productivity_change_map = {'Decrease': -1, 'No Change': 0, 'Increase': 1}
satisfaction_map = {'Unsatisfied': -1, 'Neutral': 0, 'Satisfied': 1}
physical_activity_map = {'No': 0, 'Weekly': 1, 'Daily': 2}
access_map = {'No': 0, 'Yes': 1}

df['Sleep_Quality_Ranked'] = df['Sleep_Quality'].map(sleep_quality_map)
df['Stress_Level_Ranked'] = df['Stress_Level'].map(stress_level_map)
df['Productivity_Change_Ranked'] = df['Productivity_Change'].map(productivity_change_map)
df['Satisfaction_with_Remote_Work_Ranked'] = df['Satisfaction_with_Remote_Work'].map(satisfaction_map)
df['Physical_Activity_Ranked'] = df['Physical_Activity'].map(physical_activity_map)
df['Access_to_Mental_Health_Resources_Ranked'] = df['Access_to_Mental_Health_Resources'].map(access_map)

# Encode Work_Location
encoderWL = LabelEncoder()
df['Work_Location_Encoded'] = encoderWL.fit_transform(df['Work_Location'])

# 3. Create BINARY Target (Burnout vs Not Burnout)
# "Burnout" = 1, Everything else (Anxiety, Depression, None) = 0
df['Target'] = df['Mental_Health_Condition'].apply(lambda x: 1 if x == 'Burnout' else 0)

# Select Features
feature_cols = [
    'Age', 'Years_of_Experience', 'Hours_Worked_Per_Week', 'Number_of_Virtual_Meetings',
    'Work_Life_Balance_Rating', 'Social_Isolation_Rating', 'Company_Support_for_Remote_Work',
    'Sleep_Quality_Ranked', 'Stress_Level_Ranked', 'Productivity_Change_Ranked',
    'Satisfaction_with_Remote_Work_Ranked', 'Access_to_Mental_Health_Resources_Ranked',
    'Physical_Activity_Ranked', 'Work_Location_Encoded'
]

X = df[feature_cols]
y = df['Target']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train Model - Gradient Boosting
model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['Not Burnout', 'Burnout']))

# Save Artifacts
joblib.dump(model, "burnout_model.pkl")
joblib.dump(encoderWL, "work_location_encoder.pkl")
print("Model saved.")

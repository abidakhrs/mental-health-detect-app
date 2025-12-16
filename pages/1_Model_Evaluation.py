import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, roc_curve, accuracy_score, matthews_corrcoef
)
from sklearn.preprocessing import LabelEncoder
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

st.set_page_config(page_title="Model Evaluation", layout="centered")
st.title("📊 Burnout Prediction Model Evaluation")

# Model selection
model_option = st.selectbox("Select a model to evaluate:", ["Gradient Boosting", "Random Forest"])

# Load encoder and data
encoderMS = LabelEncoder()
encoderWL = LabelEncoder()

df = pd.read_csv("Impact_of_Remote_Work_on_Mental_Health.csv")

# Preprocessing (same as training)
sleep_quality_map = {'Poor': 1, 'Average': 2, 'Good': 3}
stress_level_map = {'Low': 1, 'Medium': 2, 'High': 3}
productivity_change_map = {'Decrease': -1, 'No Change': 0, 'Increase': 1}
satisfaction_map = {'Unsatisfied': -1, 'Neutral': 0, 'Satisfied': 1}
physical_activity_map = {'No': 0, 'Weekly': 1, 'Daily': 2}
Access_map = {'No': 0, 'Yes': 1}

df['Stress_Level_Ranked'] = df['Stress_Level'].map(stress_level_map)
df['Sleep_Quality_Ranked'] = df['Sleep_Quality'].map(sleep_quality_map)
df['Productivity_Change_Ranked'] = df['Productivity_Change'].map(productivity_change_map)
df['Satisfaction_with_Remote_Work_Ranked'] = df['Satisfaction_with_Remote_Work'].map(satisfaction_map)
df['Access_to_Mental_Health_Resources_Ranked'] = df['Access_to_Mental_Health_Resources'].map(Access_map)
df['Physical_Activity'] = df['Physical_Activity'].fillna('No')
df['Physical_Activity_Ranked'] = df['Physical_Activity'].map(physical_activity_map)
# Binary Target: Burnout (1) vs Others (0)
df['Target'] = df['Mental_Health_Condition'].apply(lambda x: 1 if x == 'Burnout' else 0)
y = df['Target']
target_names = ['Not Burnout', 'Burnout']

df['Work_Location_Encoded'] = encoderWL.fit_transform(df['Work_Location'])
 
# Use features available in dataset
feature_cols = [
    'Age', 'Years_of_Experience', 'Hours_Worked_Per_Week', 'Number_of_Virtual_Meetings',
    'Work_Life_Balance_Rating', 'Social_Isolation_Rating', 'Company_Support_for_Remote_Work',
    'Sleep_Quality_Ranked', 'Stress_Level_Ranked', 'Productivity_Change_Ranked',
    'Satisfaction_with_Remote_Work_Ranked', 'Access_to_Mental_Health_Resources_Ranked',
    'Physical_Activity_Ranked', 'Work_Location_Encoded'
]

for col in df.columns:
    mode_value = df[col].mode()[0]
    df[col] = df[col].fillna(mode_value)

joblib.dump(encoderMS, "state_encoder.pkl")

X = df[feature_cols]

# Split Data to ensure realistic evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Logistic Regression Correlation Coefficient (Feature Selection) - REMOVED or UPDATED
# Since we removed Logistic Regression from the strict options, we can either keep it as separate analysis or remove.
# For simplicity, let's remove the visual correlation block or adjust it to simple correlation matrix of features vs target.

st.subheader("📈 Feature Correlation with Target")
numeric_cols = feature_cols + ['Target']    
corr_matrix = df[numeric_cols].corr()
target_corr = corr_matrix['Target'].drop('Target')

st.subheader("🔢 Correlation Coefficient Table")
st.dataframe(target_corr.sort_values(ascending=False).rename("Correlation"))

st.subheader("📊 Correlation Bar Chart")
fig, ax = plt.subplots()
sns.barplot(x=target_corr.values, y=target_corr.index, ax=ax)
ax.set_title("Correlation with Burnout Target")
ax.set_xlabel("Correlation Coefficient")
st.pyplot(fig)

# Model Training
if model_option == "Gradient Boosting":
    model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42)
    model.fit(X_train, y_train)
elif model_option == "Random Forest":
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)

# Predict and Evaluate on Test Set
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
st.subheader("✅ Accuracy (Test Set)")
st.metric(label="Model Accuracy", value=f"{accuracy:.2f}")

mcc = matthews_corrcoef(y_test, y_pred)
st.subheader("📐 Matthews Correlation Coefficient")
st.write(f"MCC Score: **{mcc:.2f}**")

report_dict = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
st.subheader("📄 Classification Report")
st.dataframe(report_df)

st.subheader("🔁 Confusion Matrix")
fig_cm, ax_cm = plt.subplots()
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=target_names, ax=ax_cm)
st.pyplot(fig_cm)

if len(target_names) == 2:
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)

    st.subheader("📈 ROC Curve")
    fig_roc, ax_roc = plt.subplots()
    ax_roc.plot(fpr, tpr, label=f"ROC AUC = {auc:.2f}")
    ax_roc.plot([0, 1], [0, 1], 'k--')
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curve")
    ax_roc.legend()
    st.pyplot(fig_roc)

if model_option == "Random Forest":
    st.subheader("📌 Feature Importance (Random Forest)")
    importances = model.feature_importances_
    features = X.columns
    importance_df = pd.DataFrame({"Feature": features, "Importance": importances})
    importance_df = importance_df.sort_values("Importance", ascending=False)

    fig_imp, ax_imp = plt.subplots()
    sns.barplot(x="Importance", y="Feature", data=importance_df, ax=ax_imp)
    ax_imp.set_title("Feature Importance (Random Forest)")
    st.pyplot(fig_imp)
    st.dataframe(importance_df.reset_index(drop=True))

if model_option in ["Random Forest", "Gradient Boosting"]:
    st.subheader(f"📌 Feature Importance ({model_option})")
    importances = model.feature_importances_
    features = X.columns
    importance_df = pd.DataFrame({"Feature": features, "Importance": importances})
    importance_df = importance_df.sort_values("Importance", ascending=False)

    fig_imp, ax_imp = plt.subplots()
    sns.barplot(x="Importance", y="Feature", data=importance_df, ax=ax_imp)
    ax_imp.set_title(f"Feature Importance ({model_option})")
    st.pyplot(fig_imp)
    st.dataframe(importance_df.reset_index(drop=True))

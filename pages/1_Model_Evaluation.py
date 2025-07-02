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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Model Evaluation", layout="centered")
st.title("📊 Burnout Prediction Model Evaluation")

# Model selection
model_option = st.selectbox("Select a model to evaluate:", ["Random Forest", "Logistic Regression"])

# Load encoder and data
encoderMS = LabelEncoder()
encoderGender = LabelEncoder()
encoderJR = LabelEncoder()
encoderIndustry = LabelEncoder()
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
df['Mental_Health_Condition'] = df['Mental_Health_Condition'].fillna('Good')
df['Mental_State'] = df['Mental_Health_Condition'].replace({
    'Burnout': 'Burnout',
    'Anxiety': 'Burnout',
    'Depression': 'Burnout',
    'None': 'Good'
})

df['Mental_State_Encoded'] = encoderMS.fit_transform(df['Mental_State'])
df['Gender_Encoded'] = encoderGender.fit_transform(df['Gender'])
df['Job_Role_Encoded'] = encoderJR.fit_transform(df['Job_Role'])
df['Industry_Encoded'] = encoderIndustry.fit_transform(df['Industry'])
df['Work_Location_Encoded'] = encoderWL.fit_transform(df['Work_Location'])

joblib.dump(encoderMS, "state_encoder.pkl")

X = df[[
    'Hours_Worked_Per_Week', 'Work_Life_Balance_Rating',
    'Sleep_Quality_Ranked', 'Stress_Level_Ranked']]
y = df['Mental_State_Encoded']

# Logistic Regression Correlation Coefficient (Feature Selection)
if model_option == "Logistic Regression":
    st.subheader("📈 Feature Correlation (Logistic Regression)")
    numeric_cols = ['Age', 'Years_of_Experience', 'Hours_Worked_Per_Week', 'Number_of_Virtual_Meetings',
                    'Work_Life_Balance_Rating', 'Social_Isolation_Rating', 'Company_Support_for_Remote_Work',
                    'Sleep_Quality_Ranked', 'Stress_Level_Ranked', 'Productivity_Change_Ranked', 'Satisfaction_with_Remote_Work_Ranked',
                    'Access_to_Mental_Health_Resources_Ranked', 'Physical_Activity_Ranked', 'Gender_Encoded', 'Job_Role_Encoded',
                    'Industry_Encoded', 'Work_Location_Encoded', 'Mental_State_Encoded']

    corr_matrix = df[numeric_cols].corr()
    target_corr = corr_matrix['Mental_State_Encoded'].drop('Mental_State_Encoded')

    st.subheader("🔢 Correlation Coefficient Table")
    st.dataframe(target_corr.sort_values(ascending=False).rename("Correlation"))

    st.subheader("📊 Correlation Bar Chart")
    fig, ax = plt.subplots()
    sns.barplot(x=target_corr.values, y=target_corr.index, ax=ax)
    ax.set_title("Correlation with Mental_State_Encoded")
    ax.set_xlabel("Correlation Coefficient")
    st.pyplot(fig)

# Model Training
if model_option == "Random Forest":
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
elif model_option == "Logistic Regression":
    model = LogisticRegression(solver='lbfgs', max_iter=1000, class_weight='balanced')
    model.fit(X, y)

# Predict and Evaluate
y_pred = model.predict(X)

accuracy = accuracy_score(y, y_pred)
st.subheader("✅ Accuracy")
st.metric(label="Model Accuracy", value=f"{accuracy:.2f}")

mcc = matthews_corrcoef(y, y_pred)
st.subheader("📐 Matthews Correlation Coefficient")
st.write(f"MCC Score: **{mcc:.2f}**")

report_dict = classification_report(y, y_pred, target_names=encoderMS.classes_, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
st.subheader("📄 Classification Report")
st.dataframe(report_df)

st.subheader("🔁 Confusion Matrix")
fig_cm, ax_cm = plt.subplots()
ConfusionMatrixDisplay.from_predictions(y, y_pred, display_labels=encoderMS.classes_, ax=ax_cm)
st.pyplot(fig_cm)

if len(encoderMS.classes_) == 2:
    y_proba = model.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y, y_proba)
    auc = roc_auc_score(y, y_proba)

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

if model_option == "Logistic Regression":
    st.subheader("📌 Feature Coefficients (Logistic Regression)")
    coef = model.coef_[0]
    coef_df = pd.DataFrame({"Feature": X.columns, "Coefficient": coef})
    coef_df['Absolute'] = coef_df['Coefficient'].abs()
    coef_df = coef_df.sort_values("Absolute", ascending=False)

    fig_coef, ax_coef = plt.subplots()
    sns.barplot(x="Coefficient", y="Feature", data=coef_df, ax=ax_coef)
    ax_coef.set_title("Logistic Regression Coefficients")
    st.pyplot(fig_coef)
    st.dataframe(coef_df.drop(columns="Absolute").reset_index(drop=True))

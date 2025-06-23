# model_evaluation_streamlit.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, roc_curve, accuracy_score, matthews_corrcoef
)
import joblib
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Model Evaluation", layout="centered")
st.title("📊 Burnout Prediction Model Evaluation")

# Model selection
model_option = st.selectbox("Select a model to evaluate:", ["Random Forest", "Logistic Regression"])

# Load encoder and data
encoder = joblib.load("state_encoder.pkl")
original_data = pd.read_csv("Impact_of_Remote_Work_on_Mental_Health.csv")

# Preprocessing (same as training)
sleep_quality_map = {'Poor': 1, 'Average': 2, 'Good': 3}
stress_level_map = {'Low': 1, 'Medium': 2, 'High': 3}

original_data['Stress_Level_Ranked'] = original_data['Stress_Level'].map(stress_level_map)
original_data['Sleep_Quality_Ranked'] = original_data['Sleep_Quality'].map(sleep_quality_map)
original_data['Mental_Health_Condition'] = original_data['Mental_Health_Condition'].fillna('Good')
original_data['Mental_State'] = original_data['Mental_Health_Condition'].replace({
    'Burnout': 'Burnout',
    'Anxiety': 'Burnout',
    'Depression': 'Burnout',
    'None': 'Good'
})

original_data['Mental_State_Encoded'] = encoder.transform(original_data['Mental_State'])

X = original_data[[
    'Hours_Worked_Per_Week', 'Work_Life_Balance_Rating',
    'Sleep_Quality_Ranked', 'Stress_Level_Ranked']]
y = original_data['Mental_State_Encoded']

# Print dataset info for analysis
st.sidebar.markdown("### Dataset Summary")
st.sidebar.write(f"📊 Total Rows: {original_data.shape[0]}")
st.sidebar.write(f"🔢 Total Features: {X.shape[1]}")

class_counts = original_data['Menpttal_State'].value_counts()
st.sidebar.markdown("### Class Distribution (Before SMOTE)")
st.sidebar.dataframe(class_counts.reset_index().rename(columns={'index': 'Mental State', 'Mental_State': 'Count'}))

# Load and evaluate the selected model
if model_option == "Random Forest":
    model = joblib.load("burnout_model.pkl")
elif model_option == "Logistic Regression":
    model = LogisticRegression(solver='lbfgs', max_iter=1000, class_weight='balanced')
    model.fit(X, y)

# Predict
y_pred = model.predict(X)

# Accuracy
accuracy = accuracy_score(y, y_pred)
st.subheader("✅ Accuracy")
st.metric(label="Model Accuracy", value=f"{accuracy:.2f}")

# Matthews Correlation Coefficient
mcc = matthews_corrcoef(y, y_pred)
st.subheader("📐 Matthews Correlation Coefficient")
st.write(f"MCC Score: **{mcc:.2f}**")

# Classification Report
report_dict = classification_report(y, y_pred, target_names=encoder.classes_, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
st.subheader("📄 Classification Report")
st.dataframe(report_df)

# Confusion Matrix
st.subheader("🔁 Confusion Matrix")
fig_cm, ax_cm = plt.subplots()
ConfusionMatrixDisplay.from_predictions(y, y_pred, display_labels=encoder.classes_, ax=ax_cm)
st.pyplot(fig_cm)

# ROC Curve (for binary classification)
if len(encoder.classes_) == 2:
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

# Feature Importance (only for Random Forest)
if model_option == "Random Forest":
    st.subheader("📌 Feature Importance")
    importances = model.feature_importances_
    features = X.columns
    importance_df = pd.DataFrame({"Feature": features, "Importance": importances})
    importance_df = importance_df.sort_values("Importance", ascending=False)

    fig_imp, ax_imp = plt.subplots()
    sns.barplot(x="Importance", y="Feature", data=importance_df, ax=ax_imp)
    ax_imp.set_title("Feature Importance (Random Forest)")
    st.pyplot(fig_imp)

    st.dataframe(importance_df.reset_index(drop=True))

# Feature Coefficients (for Logistic Regression)
if model_option == "Logistic Regression":
    st.subheader("📌 Feature Importance (Logistic Regression Coefficients)")
    coef = model.coef_[0]
    coef_df = pd.DataFrame({"Feature": X.columns, "Coefficient": coef})
    coef_df['Absolute'] = coef_df['Coefficient'].abs()
    coef_df = coef_df.sort_values("Absolute", ascending=False)

    fig_coef, ax_coef = plt.subplots()
    sns.barplot(x="Coefficient", y="Feature", data=coef_df, ax=ax_coef)
    ax_coef.set_title("Logistic Regression Coefficients")
    st.pyplot(fig_coef)

    st.dataframe(coef_df.drop(columns="Absolute").reset_index(drop=True))

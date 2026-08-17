import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Breast Cancer Malignancy Predictor",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Breast Cancer Malignancy Predictor")
st.write(
    "This app predicts breast tumor malignancy. Use the **Single Patient Prediction** "
    "tab for interactive slider-based predictions, or the **Batch Evaluation** tab to "
    "upload a test dataset and view model performance."
)

FEATURE_COLS = None


# ---------------------------------------------------------------------------
# Load dataset + train/load all models (cached so this runs only once)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_data_and_models():
    df = pd.read_csv('data.csv')
    df = df.drop(columns=['Unnamed: 32'], errors='ignore')

    X = df.drop(columns=['id', 'diagnosis'], errors='ignore')
    y = df['diagnosis'].map({'M': 1, 'B': 0})

    feature_cols = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {}

    # Pre-trained Random Forest, loaded from the saved pkl (as per project spec)
    models['Random Forest'] = joblib.load('cancer_model.pkl')

    # Remaining models trained here (same split as README) to power the dropdown
    log_reg = LogisticRegression(max_iter=5000)
    log_reg.fit(X_train, y_train)
    models['Logistic Regression'] = log_reg

    dtree = DecisionTreeClassifier(random_state=42)
    dtree.fit(X_train, y_train)
    models['Decision Tree'] = dtree

    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    models['K-Nearest Neighbors'] = knn

    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    models['Gaussian Naive Bayes'] = gnb

    return models, X, feature_cols


models, X_features, FEATURE_COLS = load_data_and_models()

# ---------------------------------------------------------------------------
# Sidebar: model selection dropdown (applies to both tabs)
# ---------------------------------------------------------------------------
st.sidebar.header("⚙️ Model Selection")
selected_model_name = st.sidebar.selectbox(
    "Choose a classification model",
    list(models.keys()),
    index=0
)
selected_model = models[selected_model_name]

tab1, tab2 = st.tabs(["🔍 Single Patient Prediction", "📊 Batch Evaluation on Test Data"])

# ---------------------------------------------------------------------------
# TAB 1: Single patient prediction (original slider-based UI)
# ---------------------------------------------------------------------------
with tab1:
    st.sidebar.header("🔬 Patient Feature Inputs")

    user_inputs = {}
    for col in FEATURE_COLS:
        min_val = float(X_features[col].min())
        max_val = float(X_features[col].max())
        mean_val = float(X_features[col].mean())

        user_inputs[col] = st.sidebar.slider(
            label=col,
            min_value=min_val,
            max_value=max_val,
            value=mean_val,
            step=(max_val - min_val) / 100
        )

    input_df = pd.DataFrame([user_inputs])[FEATURE_COLS]

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Selected Input Summary")
        st.dataframe(input_df)

    with col2:
        st.subheader("Prediction Result")
        st.caption(f"Using model: **{selected_model_name}**")
        if st.button("Predict Malignancy", type="primary"):
            prediction = selected_model.predict(input_df)[0]
            probabilities = selected_model.predict_proba(input_df)[0]

            benign_prob = probabilities[0] * 100
            malignant_prob = probabilities[1] * 100

            if prediction == 1:
                st.error("### Result: Malignant ⚠️")
                st.metric(label="Malignant Probability", value=f"{malignant_prob:.2f}%")
            else:
                st.success("### Result: Benign ✅")
                st.metric(label="Benign Probability", value=f"{benign_prob:.2f}%")

            st.write("---")
            st.write("**Confidence Breakdown:**")
            st.write(f"- **Benign:** {benign_prob:.2f}%")
            st.write(f"- **Malignant:** {malignant_prob:.2f}%")

# ---------------------------------------------------------------------------
# TAB 2: Dataset upload + evaluation metrics + confusion matrix
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Upload Test Dataset")
    st.write(
        "Upload a **test-set CSV** (feature columns + an actual diagnosis column). "
        "Since Streamlit's free tier has limited capacity, please upload only the "
        "held-out **test data** (e.g. `test_data_csv___Random_forest_.csv`) rather "
        "than the full training dataset."
    )

    uploaded_file = st.file_uploader("Upload test data CSV", type=["csv"])

    def extract_true_labels(df):
        """Try common label column names/formats and return a 0/1 Series, or None."""
        if 'diagnosis' in df.columns:
            return df['diagnosis'].map({'M': 1, 'B': 0})
        if 'Actual_Diagnosis' in df.columns:
            return df['Actual_Diagnosis'].astype(int)
        if 'Actual_Label' in df.columns:
            return df['Actual_Label'].map({'Malignant': 1, 'Benign': 0})
        return None

    if uploaded_file is not None:
        test_df = pd.read_csv(uploaded_file)

        missing_cols = [c for c in FEATURE_COLS if c not in test_df.columns]
        if missing_cols:
            st.error(f"Uploaded file is missing required feature columns: {missing_cols}")
        else:
            X_uploaded = test_df[FEATURE_COLS]
            y_true = extract_true_labels(test_df)

            st.success(f"Loaded {len(test_df)} rows. Evaluating using **{selected_model_name}**.")
            st.dataframe(test_df.head())

            y_pred = selected_model.predict(X_uploaded)
            y_proba = (
                selected_model.predict_proba(X_uploaded)[:, 1]
                if hasattr(selected_model, "predict_proba")
                else None
            )

            if y_true is not None:
                # ---- b/c/d: metrics + confusion matrix + classification report ----
                st.subheader("📈 Evaluation Metrics")
                acc = accuracy_score(y_true, y_pred)
                prec = precision_score(y_true, y_pred, zero_division=0)
                rec = recall_score(y_true, y_pred, zero_division=0)
                f1 = f1_score(y_true, y_pred, zero_division=0)
                mcc = matthews_corrcoef(y_true, y_pred)
                auc = roc_auc_score(y_true, y_proba) if y_proba is not None else None

                m1, m2, m3, m4, m5, m6 = st.columns(6)
                m1.metric("Accuracy", f"{acc*100:.2f}%")
                m2.metric("Precision", f"{prec*100:.2f}%")
                m3.metric("Recall", f"{rec*100:.2f}%")
                m4.metric("F1 Score", f"{f1*100:.2f}%")
                m5.metric("MCC", f"{mcc*100:.2f}%")
                m6.metric("AUC", f"{auc*100:.2f}%" if auc is not None else "N/A")

                st.subheader("🧩 Confusion Matrix")
                cm = confusion_matrix(y_true, y_pred)
                fig, ax = plt.subplots(figsize=(4, 3.5))
                ax.imshow(cm, cmap="Blues")
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        ax.text(
                            j, i, str(cm[i, j]),
                            ha="center", va="center",
                            color="white" if cm[i, j] > cm.max() / 2 else "black",
                            fontsize=14
                        )
                ax.set_xticks([0, 1]); ax.set_xticklabels(["Benign", "Malignant"])
                ax.set_yticks([0, 1]); ax.set_yticklabels(["Benign", "Malignant"])
                ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
                ax.set_title(f"Confusion Matrix — {selected_model_name}")
                st.pyplot(fig)

                st.subheader("📋 Classification Report")
                report = classification_report(
                    y_true, y_pred,
                    target_names=["Benign", "Malignant"],
                    zero_division=0
                )
                st.text(report)
            else:
                st.warning(
                    "No ground-truth diagnosis column found (expected `diagnosis`, "
                    "`Actual_Diagnosis`, or `Actual_Label`). Predictions are shown below, "
                    "but metrics and the confusion matrix require true labels to compare against."
                )
                result_df = test_df.copy()
                result_df["Predicted"] = pd.Series(y_pred).map({1: "Malignant", 0: "Benign"})
                st.dataframe(result_df)
    else:
        st.info("👆 Upload a test CSV file to see evaluation metrics, confusion matrix, and classification report.")

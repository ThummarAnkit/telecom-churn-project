import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# --- CONFIGURATION & GLOBAL LOAD ---
MODELS_DIR = 'models'

# Use st.cache_resource to load heavy model components only once
@st.cache_resource
def load_deployment_components():
    """Load all model components (model, imputer, scaler, feature_names)."""
    try:
        # Load the best model (XGBoost)
        model = joblib.load(os.path.join(MODELS_DIR, 'churn_prediction_model.pkl'))
        # Load necessary preprocessors
        imputer = joblib.load(os.path.join(MODELS_DIR, 'imputer.pkl'))
        scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
        # Load feature names for alignment
        feature_names = joblib.load(os.path.join(MODELS_DIR, 'feature_names.pkl'))
        st.sidebar.success("Model components loaded.")
        return model, imputer, scaler, feature_names
    except Exception as e:
        st.sidebar.error(f"FATAL ERROR: Could not load model components. Please ensure all four files are saved in the 'models/' directory. Error: {e}")
        return None, None, None, None

MODEL, IMPUTER, SCALER, FEATURE_NAMES = load_deployment_components()

# --- Streamlit App Structure ---
st.title("📞 Evoastra Telecom Churn Risk Detector 💸")
st.markdown("*A model built on advanced time-series features to predict operator churn.*")

# Upload CSV File
uploaded_file = st.file_uploader("📂 Upload Telecom Data for Prediction (CSV)", type=["csv"])

if MODEL is None:
    st.warning("Prediction requires the model components to be loaded. Please check the sidebar for errors.")
elif uploaded_file is not None:
    # --- Data Loading and Initial Display ---
    df_raw = pd.read_csv(uploaded_file)
    st.write("📌 *Available columns in dataset:*", df_raw.columns.tolist())
    st.write("### 🔍 Data Preview (First 5 Rows)")
    st.dataframe(df_raw.head())
    st.write(f"📊 Dataset Shape: {df_raw.shape}")
    
    # --- Preprocessing Pipeline for Prediction ---
    try:
        # 1. Feature Alignment (CRITICAL)
        # Reindex the input DataFrame to match the 113 columns the model was trained on
        X_aligned = df_raw.reindex(columns=FEATURE_NAMES, fill_value=0)
        
        # 2. Imputation
        X_imputed = IMPUTER.transform(X_aligned)
        
        # 3. Scaling
        X_scaled = SCALER.transform(X_imputed)

    except Exception as e:
        st.error(f"❌ Error during Data Preprocessing. Ensure your uploaded data contains the necessary raw and engineered features. Error: {e}")
        st.stop()

    # --- Prediction and Evaluation ---
    st.write("### 🚀 Running Prediction...")
    
    # Predict
    predictions = MODEL.predict(X_scaled)
    probabilities = MODEL.predict_proba(X_scaled)[:, 1]
    
    # Add results back to the original DataFrame
    df_raw['predicted_churn'] = predictions
    df_raw['churn_probability'] = probabilities
    
    st.success("✅ Prediction Complete.")
    
    # --- Model Evaluation (Conditional) ---
    if 'churn_binary' in df_raw.columns:
        st.write("### 🎯 Model Evaluation (Ground Truth Present)")
        
        # Use a small sample for X_test just for displaying feature names
        # In a real app, the ground truth is often split. Here we use the whole uploaded set.
        y_true = df_raw['churn_binary'].astype(int)
        
        accuracy = accuracy_score(y_true, predictions)
        st.write(f"#### Overall Accuracy: {accuracy:.2%}")

        # Classification Report
        report = classification_report(y_true, predictions, output_dict=True, zero_division=0)
        st.write("#### 📋 Classification Report")
        st.dataframe(pd.DataFrame(report).transpose())

        # 🧾 Label Meaning Table (Adjusted for Telecom Churn)
        st.write("#### 🧾 Label Meaning")
        label_meaning = pd.DataFrame({
            "Label": [0, 1],
            "Meaning": [
                "❌ No Churn (Predicted Stable)",
                "✅ Churn (Predicted At-Risk)"
            ]
        })
        st.table(label_meaning)
    else:
        st.info("Evaluation metrics (Classification Report, Accuracy) are available only if the uploaded file contains the ground truth column named **churn\_binary**.")

    # --- Feature Importance Plot ---
    if hasattr(MODEL, "feature_importances_"):
        st.write("### 📈 Top Feature Importances")
        importances = MODEL.feature_importances_
        # Use the aligned feature names
        feature_names = FEATURE_NAMES 
        top_indices = np.argsort(importances)[-10:]

        plt.figure(figsize=(10, 6))
        plt.barh(range(len(top_indices)), importances[top_indices], color='darkred', align='center')
        plt.yticks(range(len(top_indices)), [feature_names[i].replace('_', ' ').title() for i in top_indices])
        plt.xlabel('Relative Importance (Gain)')
        plt.title('Top 10 Features Driving Churn Prediction')
        st.pyplot(plt)
        plt.close() # Close plot to free memory
    
    # --- Final Results Preview ---
    st.write("### ✅ Prediction Results Preview")
    st.dataframe(df_raw[['service_provider', 'circle', 'churn_probability', 'predicted_churn']].head(10))
    st.write(f"📊 Dataset Shape: {df_raw.shape}")
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

from xgboost import XGBClassifier
from imblearn.combine import SMOTEENN

# Streamlit App Title
st.title("🏥Smart Re-admit: AI-powered Patient Re-admission Predictor🚑")

# Upload CSV File
uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.lower().str.strip()

    st.write("📌 *Available columns in dataset:*", df.columns.tolist())
    st.write("### 🔍 Data Preview")
    st.dataframe(df.head())
    st.write(f"📊 Shape before preprocessing: {df.shape}")

    # Handling missing values
    df.replace(["?", "unknown", "Unknown"], np.nan, inplace=True)
    for col in df.columns:
        if df[col].dtype == "object":
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            df[col].fillna(df[col].median(), inplace=True)

    # Ensure target column
    if 're-admitted' not in df.columns:
        st.error("❌ The dataset must contain a 're-admitted' column.")
        st.stop()

    # Encode categorical features
    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col])

    st.write("### ✅ Processed Data")
    st.dataframe(df.head())

    X = df.drop(columns=['re-admitted'])
    y = df['re-admitted']

    # Feature Scaling
    st.write("### ⚙ Scaling Features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Balance Classes using SMOTEENN
    st.write("### 🔁 Balancing with SMOTEENN...")
    smoteenn = SMOTEENN(random_state=42)
    X_bal, y_bal = smoteenn.fit_resample(X_scaled, y)

    # Display class balance
    st.write("✅ Class Distribution After Balancing:")
    unique, counts = np.unique(y_bal, return_counts=True)
    st.write(dict(zip(unique, counts)))

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_bal, y_bal, test_size=0.2, random_state=42)

    # Compute class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

    if st.button("🚀 Train Optimized Model"):
        # Define parameter space for tuning
        param_grid = {
            'n_estimators': [100, 200, 300, 400, 500],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7, 9],
            'min_child_weight': [1, 3, 5, 7],
            'gamma': [0, 0.1, 0.2, 0.3],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.01, 0.1, 1],
            'reg_lambda': [0, 0.01, 0.1, 1]
        }

        st.write("⏳ Performing Hyperparameter Tuning (RandomizedSearchCV)...")
        base_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42, scale_pos_weight=class_weight_dict)
        grid_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_grid,
            n_iter=50,
            scoring='accuracy',
            cv=3,
            verbose=1,
            n_jobs=-1,
            random_state=42
        )
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        predictions = best_model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        st.write(f"### 🎯 Model Accuracy: {accuracy:.2%}")

        report = classification_report(y_test, predictions, output_dict=True)
        st.write("### 📋 Classification Report")
        st.dataframe(pd.DataFrame(report).transpose())

        # 🧾 Label Meaning Table
        st.write("### 🧾 Label Meaning")
        label_meaning = pd.DataFrame({
            "Label": [1, 2],
            "Meaning": [
                "✅ Yes - Patient was re-admitted after >30 days",
                "❌ No - Patient was not re-admitted"
            ]
        })
        st.table(label_meaning)

        # Feature importance plot
        st.write("### 📈 Top Feature Importances")
        importances = best_model.feature_importances_
        feature_names = X.columns
        top_indices = np.argsort(importances)[-10:]

        plt.figure(figsize=(10, 6))
        plt.barh(range(len(top_indices)), importances[top_indices], color='skyblue', align='center')
        plt.yticks(range(len(top_indices)), [feature_names[i] for i in top_indices])
        plt.xlabel('Relative Importance')
        plt.title('Top 10 Features')
        st.pyplot(plt)
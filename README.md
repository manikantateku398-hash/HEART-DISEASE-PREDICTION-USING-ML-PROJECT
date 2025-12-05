"""
Heart Disease Prediction — Full Project Script
File: heart_disease_project.py

What this does:
- Loads a local 'heart.csv' dataset (place it in same folder). If not found, it shows instructions.
- Performs EDA, preprocessing, trains multiple ML models (LogisticRegression, RandomForest, XGBoost if available, SVC),
  evaluates them (accuracy, precision, recall, f1, ROC AUC), plots results, and saves the best model.
- Generates a minimal Streamlit app file 'streamlit_app.py' that you can run to test predictions in a UI.

How to use:
1. Put your dataset file named 'heart.csv' in the same folder. The expected target column name is 'target' (0/1)
   — if your file uses a different name, update the `TARGET_COL` variable below.
2. Install dependencies: pip install -r requirements.txt
   Example requirements.txt content (you can create this file or install manually):
       pandas
       numpy
       scikit-learn
       matplotlib
       seaborn
       streamlit
       xgboost
       joblib
3. Run: python heart_disease_project.py
   This will train models and save 'best_model.joblib' and create 'streamlit_app.py'.
4. To run the Streamlit UI: streamlit run streamlit_app.py

Notes:
- The script is designed to be easy-to-follow and modular so you can copy parts into a notebook.
- If you want a Jupyter Notebook instead, copy this script into a new .ipynb cell-by-cell.

"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, classification_report)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

import matplotlib.pyplot as plt
import seaborn as sns

# Optional XGBoost
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

# Model save
try:
    import joblib
except Exception:
    from sklearn.externals import joblib


# ----------------------------
# Configuration
# ----------------------------
DATA_PATH = 'heart.csv'   # place your dataset in the same folder with this name
TARGET_COL = 'target'     # change if your target column is named differently
RANDOM_STATE = 42
TEST_SIZE = 0.2

# ----------------------------
# Utility functions
# ----------------------------

def load_data(path=DATA_PATH):
    if not os.path.exists(path):
        msg = (
            f"Dataset file '{path}' not found in current directory.\n"
            "Please download a heart disease dataset and save as 'heart.csv' in this folder.\n"
            "Common sources: UCI Machine Learning Repository or Kaggle (search 'heart disease dataset').\n"
            "Alternatively, prepare a CSV with clinical features and a binary target column named 'target'.\n"
        )
        raise FileNotFoundError(msg)
    df = pd.read_csv(path)
    return df


def basic_eda(df):
    print('\n=== DATA SHAPE ===')
    print(df.shape)
    print('\n=== HEAD ===')
    print(df.head())
    print('\n=== INFO ===')
    print(df.info())
    print('\n=== MISSING VALUES ===')
    print(df.isnull().sum())

    # target distribution
    if TARGET_COL in df.columns:
        print('\n=== TARGET DISTRIBUTION ===')
        print(df[TARGET_COL].value_counts(normalize=True))

    # quick histograms
    df.hist(figsize=(12,10))
    plt.tight_layout()
    plt.show()


def preprocess(df, scale=True):
    # Simple preprocessing pipeline: handle categorical if present, fillna (if needed), scale numeric
    df_copy = df.copy()

    # Basic missing value handling: fill numeric with median, categorical with mode
    for col in df_copy.columns:
        if df_copy[col].isnull().sum() > 0:
            if df_copy[col].dtype in [np.float64, np.int64]:
                df_copy[col] = df_copy[col].fillna(df_copy[col].median())
            else:
                df_copy[col] = df_copy[col].fillna(df_copy[col].mode()[0])

    # Separate X, y
    if TARGET_COL not in df_copy.columns:
        raise KeyError(f"Target column '{TARGET_COL}' not found in dataframe columns: {df_copy.columns.tolist()}")

    X = df_copy.drop(columns=[TARGET_COL])
    y = df_copy[TARGET_COL]

    # One-hot encode categorical columns (if any object or category dtype)
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    # Scaling
    scaler = None
    if scale:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X = pd.DataFrame(X_scaled, columns=X.columns)

    return X, y, scaler


def evaluate_model(model, X_test, y_test, verbose=True):
    preds = model.predict(X_test)
    probs = None
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X_test)[:,1]
    elif hasattr(model, 'decision_function'):
        try:
            probs = model.decision_function(X_test)
        except Exception:
            probs = None

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    roc = roc_auc_score(y_test, probs) if probs is not None else None

    if verbose:
        print('Accuracy: {:.4f}'.format(acc))
        print('Precision: {:.4f}'.format(prec))
        print('Recall: {:.4f}'.format(rec))
        print('F1-score: {:.4f}'.format(f1))
        if roc is not None:
            print('ROC AUC: {:.4f}'.format(roc))
        print('\nConfusion Matrix:')
        print(confusion_matrix(y_test, preds))
        print('\nClassification Report:')
        print(classification_report(y_test, preds))

    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'roc_auc': roc}


# ----------------------------
# Training and model selection
# ----------------------------

def train_and_select(X_train, y_train):
    # Define models
    models = {
        'LogisticRegression': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
        'RandomForest': RandomForestClassifier(random_state=RANDOM_STATE),
        'SVC': SVC(probability=True, random_state=RANDOM_STATE)
    }
    if HAS_XGB:
        models['XGBoost'] = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=RANDOM_STATE)

    results = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
        mean_score = np.mean(scores)
        print(f"{name} CV ROC AUC: {mean_score:.4f}")
        results[name] = {'model': model, 'cv_roc_auc': mean_score}

    # Pick best by CV ROC AUC
    best_name = max(results.items(), key=lambda kv: kv[1]['cv_roc_auc'])[0]
    best_model = results[best_name]['model']
    print(f"\nBest model selected: {best_name} (CV ROC AUC={results[best_name]['cv_roc_auc']:.4f})")

    return best_name, best_model, results


# ----------------------------
# Hyperparameter tuning (example for RandomForest)
# ----------------------------

def tune_random_forest(X_train, y_train):
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5]
    }
    rf = RandomForestClassifier(random_state=RANDOM_STATE)
    grid = GridSearchCV(rf, param_grid, cv=4, scoring='roc_auc', n_jobs=-1)
    grid.fit(X_train, y_train)
    print('RandomForest best params:', grid.best_params_)
    return grid.best_estimator_


# ----------------------------
# Main flow
# ----------------------------

def main():
    print('Loading data...')
    df = load_data(DATA_PATH)

    basic_eda(df)

    print('\nPreprocessing...')
    X, y, scaler = preprocess(df, scale=True)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

    print('\nTraining candidate models...')
    best_name, best_model, all_results = train_and_select(X_train, y_train)

    # Optionally tune RandomForest if it's best
    if best_name == 'RandomForest':
        print('\nTuning RandomForest hyperparameters...')
        best_model = tune_random_forest(X_train, y_train)

    print('\nEvaluating best model on test set...')
    metrics = evaluate_model(best_model, X_test, y_test)

    # Save the model and scaler
    save_dict = {'model': best_model}
    if scaler is not None:
        save_dict['scaler'] = scaler

    joblib.dump(save_dict, 'best_model.joblib')
    print("Saved best model and scaler to 'best_model.joblib'.")

    # Simple feature importance plot for tree-based models
    if hasattr(best_model, 'feature_importances_'):
        fi = best_model.feature_importances_
        fi_series = pd.Series(fi, index=X.columns).sort_values(ascending=False)[:20]
        plt.figure(figsize=(8,6))
        sns.barplot(x=fi_series.values, y=fi_series.index)
        plt.title('Top feature importances')
        plt.tight_layout()
        plt.show()

    # Create a minimal Streamlit app file
    create_streamlit_app(X.columns.tolist())
    print("Created 'streamlit_app.py' — run it with: streamlit run streamlit_app.py")


# ----------------------------
# Generate Streamlit app helper
# ----------------------------

def create_streamlit_app(feature_names):
    app_code = f"""
import streamlit as st
import joblib
import numpy as np

st.title('Heart Disease Prediction Demo')
st.write('Enter patient values and click Predict')

# Load model
obj = joblib.load('best_model.joblib')
model = obj['model']
scaler = obj.get('scaler', None)

# Input widgets
inputs = {{}}
"""

    # add numeric inputs for each feature
    for feat in feature_names:
        safe_name = feat.replace(' ', '_')
        app_code += f"inputs['{feat}'] = st.number_input('{feat}', value=0.0)\n"

    app_code += "\nif st.button('Predict'):\n"
    app_code += "    X = np.array([ [inputs[f] for f in inputs] ])\n"
    app_code += "    if scaler is not None:\n"
    app_code += "        X = scaler.transform(X)\n"
    app_code += "    pred = model.predict(X)[0]\n"
    app_code += "    prob = None\n"
    app_code += "    try:\n"
    app_code += "        prob = model.predict_proba(X)[0][1]\n"
    app_code += "    except Exception:\n"
    app_code += "        pass\n"
    app_code += "    st.write('Prediction (1=heart disease, 0=no):', int(pred))\n"
    app_code += "    if prob is not None:\n"
    app_code += "        st.write('Probability of heart disease:', float(prob))\n"

    with open('streamlit_app.py', 'w') as f:
        f.write(app_code)


# ----------------------------
# Run
# ----------------------------
if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print('ERROR:', e)
        print('\nIf you need help getting the dataset or want a notebook version, reply and I will provide it.')

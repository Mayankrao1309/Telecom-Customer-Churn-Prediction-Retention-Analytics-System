import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV

def load_and_prepare_data(path="model/churn_dataset.csv"):
    df = pd.read_csv(path)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    if 'customerID' in df.columns:
        df.drop(columns=['customerID'], inplace=True)
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    return df

def encode_features(df):
    df_copy = df.copy()
    for col in df_copy.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df_copy[col] = le.fit_transform(df_copy[col])
    return df_copy

def load_calibrated_model(model_path="model/xgboost_churn_model.joblib", data_path="model/churn_dataset.csv"):
    df = load_and_prepare_data(data_path)
    X = encode_features(df.drop('Churn', axis=1))
    y = df['Churn']

    base_model = joblib.load(model_path)
    calibrated_model = CalibratedClassifierCV(base_model, method='isotonic', cv=5)
    calibrated_model.fit(X, y)
    df['pred_prob'] = calibrated_model.predict_proba(X)[:, 1]

    return df, calibrated_model

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
# Preprocessing function







# Train the model
def train_model():
    df= pd.read_csv("diabetes.csv")
    y = df['Outcome']
    X = df.drop(columns='Outcome')

    # -------------------------------
    # Build Pipeline
    # -------------------------------

    categorical_features = ['NewBMI', 'NewInsulinScore', 'NewGlucose']
    numeric_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    from feature_engineering import FeatureEngineer


    preprocessor = Pipeline(steps=[
        ('engineer', FeatureEngineer()),
        ('transform', ColumnTransformer(transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
        ]))
    ])

    pipeline = Pipeline(steps=[
        ('preprocessing', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000))
    ])

    # -------------------------------
    # Train and Evaluate
    # -------------------------------

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)

    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train, y_train)
    import os
    # Save the model and scaler
    os.makedirs("model", exist_ok=True)
    # joblib.dump(log_reg, "model/diabetes_model.pkl")
    joblib.dump(pipeline, 'model/diabetes_pipeline_model.pkl')


if __name__ == "__main__":
    train_model()

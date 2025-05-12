import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Cap outliers
        def cap_outliers(series):
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            return np.where(series > upper, upper, np.where(series < lower, lower, series))

        X['Insulin'] = cap_outliers(X['Insulin'])
        X['BMI'] = cap_outliers(X['BMI'])
        X['Glucose'] = cap_outliers(X['Glucose'])

        # BMI Category
        def bmi_category(bmi):
            if bmi < 18.5:
                return "Underweight"
            elif 18.5 <= bmi <= 24.9:
                return "Normal"
            elif 25 <= bmi <= 29.9:
                return "Overweight"
            elif 30 <= bmi <= 34.9:
                return "Obesity 1"
            elif 35 <= bmi <= 39.9:
                return "Obesity 2"
            else:
                return "Obesity 3"

        X['NewBMI'] = X['BMI'].apply(bmi_category)
        X['NewInsulinScore'] = X['Insulin'].apply(lambda x: 'Normal' if 16 <= x <= 166 else 'Abnormal')

        def glucose_level(glucose):
            if glucose <= 70:
                return "Low"
            elif 70 < glucose <= 99:
                return "Normal"
            elif 99 < glucose <= 126:
                return "High"
            else:
                return "Very High"

        X['NewGlucose'] = X['Glucose'].apply(glucose_level)

        return X
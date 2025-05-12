# 🩺 Diabetes Dataset and Prediction Model (2025)

This folder contains all of the data and source code used for training a machine learning model to predict diabetes, based on the **Pima Indian Diabetes Dataset**. The project was built and tested in a Python environment and can be deployed using Docker.

---

## 📄 File Metadata

### diabetes.csv

**What is this?**  
This dataset contains medical diagnostic measurements of female patients of at least 21 years of age from the Pima Indian population. It is used to predict the onset of diabetes.

**Source(s) & Methods:**  
This dataset originates from the National Institute of Diabetes and Digestive and Kidney Diseases. It is available from various machine learning repositories, including Kaggle and the UCI Machine Learning Repository. It was used as-is but includes additional preprocessing and feature engineering steps in the code provided (`model.py`).

**Spatial Applicability:**  
United States (Pima Indian population, Arizona)

**Temporal Applicability:**  
Historical medical records; specific years of collection not disclosed.

**Observations (Rows):**  
Each row represents one patient's diagnostic record.

**Variables (Columns):**

| Header                     | Description                                                                                   | Data Type |
|--------------------------- |-----------------------------------------------------------------------------------------------|-----------|
| Pregnancies                | Number of times pregnant                                                                      | Integer   |
| Glucose                    | Plasma glucose concentration a 2 hours in an oral glucose tolerance test                      | Integer   |
| BloodPressure              | Diastolic blood pressure (mm Hg)                                                              | Integer   |
| SkinThickness              | Triceps skinfold thickness (mm)                                                               | Integer   |
| Insulin                    | 2-Hour serum insulin (mu U/ml)                                                                | Integer   |
| BMI                        | Body mass index (weight in kg/(height in m)^2)                                                | Float     |
| DiabetesPedigreeFunction   | Diabetes pedigree function (a function which scores likelihood of diabetes based on family)   | Float     |
| Age                        | Age in years                                                                                  | Integer   |
| Outcome                    | Binary variable indicating diabetes status (1 = diabetic, 0 = non-diabetic)                   | Integer   |

---

## 🧠 model.py

**What is this?**  
Python script that preprocesses the dataset, trains a `LogisticRegression` model, and outputs a prediction for a hardcoded input sample. It includes detailed handling of missing data, outlier treatment, feature scaling, and engineered categorical features based on BMI, glucose, and insulin values.

**Key Steps:**
- Zero values in several fields are replaced with NaN and filled using class-wise medians.
- Categorical feature engineering for:
    - BMI category (`NewBMI`)
    - Glucose category (`NewGlucose`)
    - Insulin status (`NewInsulinScore`)
- One-hot encoding of categorical features.
- Scaling using `RobustScaler` (for model training).
- A test sample is passed into the trained model for prediction.

**Requirements:**
See `requirements.txt` for all Python dependencies.

---

## 🐳 Docker Support

A Dockerfile can be used to build and run this prediction model in an isolated environment.

### Sample Docker Commands

```bash
docker-compose up --build
```
Please visit the url to predict the outcome using a json.

Sample Json:
```commandline
{
  "Pregnancies": 2,
  "Glucose": 197,
  "BloodPressure": 70,
  "SkinThickness": 45,
  "Insulin": 543,
  "BMI": 26.6,
  "DiabetesPedigreeFunction": 0.351,
  "Age": 50
}

```


# Heart Disease Prediction (ML + Flask Web App)

An end-to-end machine learning project: exploratory analysis, a scikit-learn preprocessing + model pipeline, and a **Flask web app** to get predictions from a form or a CSV file.

## Features
- **EDA & modeling notebook** (`notebooks/`): data exploration, encoding strategies (binary, ordinal, one-hot), scaling, PCA, and comparison of Logistic Regression, SVM and Random Forest.
- **Training script** (`train_pipeline.py`): builds a `ColumnTransformer` + classifier `Pipeline` and saves it with `joblib`.
- **Web app** (`app.py`):
  - `/` – input form for a single patient
  - `/predict` – returns the prediction and class probabilities as JSON
  - `/predict_csv` – upload a CSV and download predictions with disease probability

## Input features
`age, sex, chest_pain_type, resting_blood_pressure, cholestoral, fasting_blood_sugar, rest_ecg, Max_heart_rate, exercise_induced_angina, oldpeak, slope, vessels_colored_by_flourosopy, thalassemia`

## Run locally
```bash
pip install -r requirements.txt
python app.py
# open http://127.0.0.1:5000
```
To retrain the model: `python train_pipeline.py` (regenerates `heart_disease_model_pipeline.pkl`).

For anything beyond local use, set `SECRET_KEY` as an environment variable and run with `debug=False`.

## Project structure
```
app.py                          Flask app
train_pipeline.py               Model training pipeline
heart_disease_model_pipeline.pkl  Trained pipeline
HeartDiseaseTrain-Test.csv      Dataset
templates/interface.html        Web UI
notebooks/                      EDA and modeling notebook
```

> Educational project only. Not a medical diagnostic tool.

# Bank Marketing Prediction App

This repository contains a Streamlit web application that predicts whether a client will subscribe to a term deposit based on marketing data. The model used is an XGBoost classifier trained on the `bank-additional-full.csv` dataset.

## 📂 Repository Contents

- `app.py` — The main Streamlit app for uploading data and generating predictions. This generates a dataframe with the appended predictions as "predicted y".
- `xgb_model.pkl` — The trained XGBoost model.
- `model_columns.pkl` — List of model input features used during training.
- `requirements.txt` — List of Python packages required to run the app.
- `data_processing.ipynb` — Jupyter notebook containing data cleaning, feature engineering, and model training steps.
- `Report.pdf` — A report summarizing the model performance and insights.
  
---

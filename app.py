#!/usr/bin/env python
# coding: utf-8

import joblib
import streamlit as st
import pandas as pd
import numpy as np

# Load model and required columns
model = joblib.load("xgb_model.pkl")
required_columns = joblib.load("model_columns.pkl")

def preprocess_input(df):
    # Feature engineering
    df['contacted_before'] = np.where(df['pdays'] == 999, 0, 1)

    month_map = {'jan':1, 'feb':2, 'mar':3, 'apr':4, 'may':5, 'jun':6,
                 'jul':7, 'aug':8, 'sep':9, 'oct':10, 'nov':11, 'dec':12}
    df['month_num'] = df['month'].map(month_map)
    df['quarter'] = pd.to_datetime(df['month_num'], format='%m').dt.quarter

    df['campaign_vs_previous'] = df['campaign'] / (df['previous'] + 1)

    duration_bins = [0, 60, 180, np.inf]
    duration_labels = ['short', 'medium', 'long']
    df['duration_category'] = pd.cut(df['duration'], bins=duration_bins, labels=duration_labels)

    # One-hot encoding
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    df = pd.get_dummies(df, columns=['duration_category'] + categorical_cols, drop_first=True)

    # Align with training columns
    for col in required_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[required_columns]

    return df

# Streamlit UI
st.title("Bank Marketing Campaign Prediction App")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])  

if uploaded_file is not None:
    try:
        input_df = pd.read_csv(uploaded_file, sep=';')

        # Drop original 'y' column if present to show only predicted yas will happen in real use case
        if 'y' in input_df.columns:
            input_df = input_df.drop(columns=['y'])

        processed_df = preprocess_input(input_df.copy())
        predictions = model.predict(processed_df)

        # Map 0/1 to "no"/"yes"
        prediction_labels = ["yes" if p == 1 else "no" for p in predictions]

        # Add predictions
        input_df["predicted y"] = prediction_labels

        st.write("### Client Term Deposit Subscription Predictions")
        st.dataframe(input_df.reset_index(drop=True))

    except Exception as e:
        st.error(f"Error: {e}")

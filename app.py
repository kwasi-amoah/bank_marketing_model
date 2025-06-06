#!/usr/bin/env python
# coding: utf-8

# In[1]:


import joblib
import streamlit as st
import pandas as pd
import numpy as np


# In[2]:


# Load model and columns list
model = joblib.load("xgb_model.pkl")
required_columns = joblib.load("model_columns.pkl")  # same as inside preprocess_input

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

    # Align columns with training data
    for col in required_columns:
        if col not in df.columns:
            df[col] = 0  # Add missing dummy columns
    df = df[required_columns]  # Ensure correct order

    return df

st.title("XGBoost Model Prediction with Preprocessing")

uploaded_file = st.file_uploader("Upload a CSV file with input data", type=["csv"])  

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, sep=';')
    st.write("Parsed Data:")
    st.dataframe(df)

    try:
        processed_df = preprocess_input(df)
    except Exception as e:
        st.error(f"Preprocessing failed: {e}")
    else:
        st.write("### Processed Data")
        st.dataframe(processed_df)

        
        # Predict
        preds = model.predict(processed_df)

        # Map 1 -> 'yes', 0 -> 'no'
        labels = ["yes" if pred == 1 else "no" for pred in preds]
        
        # Create DataFrame with column name 'predicted y'
        pred_df = pd.DataFrame(labels, columns=["predicted y"])
        
        # Show the labeled predictions
        st.write("### Predictions")
        st.dataframe(pred_df)


       
        

        



# In[ ]:





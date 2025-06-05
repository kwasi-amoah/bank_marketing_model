#!/usr/bin/env python
# coding: utf-8

# In[1]:


import joblib
from flask import Flask, request, jsonify


# In[3]:


app = Flask(__name__)
model = joblib.load("C:/Users/User/Desktop/data 1 (4) (1)/data/bank_marketing_model/model/xgb_model.pkl")
required_columns = joblib.load("C:/Users/User/Desktop/data 1 (4) (1)/data/bank_marketing_model/model/model_columns.pkl")

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
    required_columns = joblib.load('model/columns.pkl')  # this must be saved during training
    for col in required_columns:
        if col not in df.columns:
            df[col] = 0  # Add missing dummy columns
    df = df[required_columns]  # Ensure correct order

    return df


@app.route('/')
def home():
    return "Bank Marketing Prediction Model is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_df = pd.DataFrame([data])
        processed_df = preprocess_input(input_df)
        prediction = model.predict(processed_df)[0]
        return jsonify({'prediction': int(prediction)})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    import os
    os.environ['FLASK_ENV'] = 'development'
    app.run(debug=True, use_reloader=False)


# In[ ]:





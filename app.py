# app.py
import streamlit as st
import pandas as pd
import pickle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Configuration
MODEL_PATH = 'clv_model.pkl'
SCALER_PATH = 'scaler.pkl'

# Page setup
st.set_page_config(
    page_title="Customer Lifetime Value Predictor",
    page_icon="💰",
    layout="wide"
)

# Load model
@st.cache_resource
def load_model():
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model()


# Sidebar
st.sidebar.title("About")

st.sidebar.info(
    """
     **Project Created By:**
     Dev Krishan  
     Arya college of Engineering and IT 
    
    
    This application predicts Customer Lifetime Value (CLV) based on:
    - Recency (days since last purchase)
    - Frequency (number of transactions)
    - Monetary value (total spend)
    - Average order value
    - Customer tenure
    """
)
st.sidebar.image('feature_importance.png', caption='Feature Importance')

# Main content
st.title("💰 Customer Lifetime Value Predictor")
st.markdown("""
Predict the future value of a customer to your business based on their purchase history.
""")

# Input form
col1, col2 = st.columns(2)

with col1:
    st.subheader("Customer Details")
    recency = st.number_input(
        'Days since last purchase (Recency)',
        min_value=0,
        max_value=365*5,
        value=30
    )
    frequency = st.number_input(
        'Number of transactions (Frequency)',
        min_value=1,
        max_value=1000,
        value=5
    )
    
with col2:
    st.subheader("Purchase History")
    monetary = st.number_input(
        'Total spend (Monetary Value)',
        min_value=0.0,
        max_value=1000000.0,
        value=500.0
    )
    avg_order = st.number_input(
        'Average order value',
        min_value=0.0,
        max_value=10000.0,
        value=100.0
    )
    tenure = st.number_input(
        'Days as customer (Tenure)',
        min_value=0,
        max_value=365*10,
        value=365
    )

# Prediction
if st.button('Predict Customer Lifetime Value'):
    input_data = pd.DataFrame([[recency, frequency, monetary, avg_order, tenure]],
                            columns=['Recency', 'Frequency', 'MonetaryValue', 'AvgOrderValue', 'Tenure'])
    
    prediction = model.predict(input_data)[0]
    
    st.success(f"### Predicted Customer Lifetime Value: ${prediction:,.2f}")
    
    # Explanation
    st.subheader("How to interpret this result")
    st.markdown(f"""
    - This customer is predicted to generate **${prediction:,.2f}** in revenue over their lifetime
    - Based on similar customers with:
        - {recency} days since last purchase
        - {frequency} total transactions
        - ${monetary:,.2f} total spend
        - ${avg_order:,.2f} average order value
        - {tenure} days as a customer
    """)

# Add some sample predictions
st.markdown("---")
st.subheader("Example Customer Profiles")

examples = [
    ["New High-Value", 30, 5, 2000, 400, 180],
    ["Loyal Regular", 15, 50, 5000, 100, 730],
    ["At-Risk", 120, 10, 800, 80, 365]
]

for name, r, f, m, a, t in examples:
    example_data = pd.DataFrame([[r, f, m, a, t]],
                              columns=['Recency', 'Frequency', 'MonetaryValue', 'AvgOrderValue', 'Tenure'])
    pred = model.predict(example_data)[0]
    
    st.write(f"""
    **{name} Customer**:
    - Recency: {r} days | Frequency: {f} | Spend: ${m:,.2f}
    - Avg Order: ${a:,.2f} | Tenure: {t} days
    - **Predicted CLV**: ${pred:,.2f}
    """)

# Add feature distributions
st.markdown("---")
st.subheader("Feature Distributions")
st.image('feature_distributions.png')
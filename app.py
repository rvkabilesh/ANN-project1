import streamlit as st
import pandas as pd
import pickle
from tensorflow.keras.models import load_model

# Load pre-trained components (same as exp.py saves)
@st.cache_resource
def load_assets():
    model = load_model('ann_model.h5')
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('label_encoders.pkl', 'rb') as f:
        label_encoders = pickle.load(f)
    return model, scaler, label_encoders

model, scaler, label_encoders = load_assets()

# Feature list (same as exp.py)
features = [
    'Days for shipment (scheduled)',
    'Shipping Mode',
    'Order City',
    'Order Profit Per Order',
    'Benefit per order',
    'Order State',
    'Customer Zipcode',
    'Customer City',
    'Order Item Discount',
    'Sales per customer',
    'Order Item Total',
    'Order Country'
]

st.title("🚚 Supply Chain Late Delivery Risk Predictor")
st.write("Enter order details to predict if the shipment will be late.")

# Load dataset for dropdown options
@st.cache_data
def load_dropdown_data():
    df = pd.read_csv('DataCoSupplyChainDataset.csv', encoding='ISO-8859-1')
    return df

df_options = load_dropdown_data()

# Create input UI in two columns
col1, col2 = st.columns(2)
user_inputs = {}

with col1:
    # Left column inputs
    user_inputs['Days for shipment (scheduled)'] = st.number_input(
        'Days for shipment (scheduled)', 
        min_value=0, max_value=10, value=4
    )
    user_inputs['Shipping Mode'] = st.selectbox(
        'Shipping Mode',
        sorted(df_options['Shipping Mode'].dropna().unique())
    )
    user_inputs['Order City'] = st.selectbox(
        'Order City',
        sorted(df_options['Order City'].dropna().unique())[:100]  # Limit to first 100 for UI
    )
    user_inputs['Order Profit Per Order'] = st.number_input(
        'Order Profit Per Order',
        min_value=float(df_options['Order Profit Per Order'].min()),
        max_value=float(df_options['Order Profit Per Order'].max()),
        value=float(df_options['Order Profit Per Order'].mean()),
        step=1.0
    )
    user_inputs['Benefit per order'] = st.number_input(
        'Benefit per order',
        min_value=float(df_options['Benefit per order'].min()),
        max_value=float(df_options['Benefit per order'].max()),
        value=float(df_options['Benefit per order'].mean()),
        step=1.0
    )
    user_inputs['Order State'] = st.selectbox(
        'Order State',
        sorted(df_options['Order State'].dropna().unique())[:100]
    )

with col2:
    # Right column inputs
    user_inputs['Customer Zipcode'] = st.selectbox(
        'Customer Zipcode',
        sorted(df_options['Customer Zipcode'].fillna('Unknown').astype(str).unique())[:100]
    )
    user_inputs['Customer City'] = st.selectbox(
        'Customer City',
        sorted(df_options['Customer City'].dropna().unique())
    )
    user_inputs['Order Item Discount'] = st.number_input(
        'Order Item Discount',
        min_value=float(df_options['Order Item Discount'].min()),
        max_value=float(df_options['Order Item Discount'].max()),
        value=float(df_options['Order Item Discount'].mean()),
        step=0.1
    )
    user_inputs['Sales per customer'] = st.number_input(
        'Sales per customer',
        min_value=float(df_options['Sales per customer'].min()),
        max_value=float(df_options['Sales per customer'].max()),
        value=float(df_options['Sales per customer'].mean()),
        step=1.0
    )
    user_inputs['Order Item Total'] = st.number_input(
        'Order Item Total',
        min_value=float(df_options['Order Item Total'].min()),
        max_value=float(df_options['Order Item Total'].max()),
        value=float(df_options['Order Item Total'].mean()),
        step=1.0
    )
    user_inputs['Order Country'] = st.selectbox(
        'Order Country',
        sorted(df_options['Order Country'].dropna().unique())
    )

# Prediction Logic (following exp.py preprocessing)
if st.button("Predict Risk"):
    # 1. Create DataFrame from user inputs
    X = pd.DataFrame([user_inputs], columns=features)
    
    # 2. Fill missing values (same as exp.py)
    X = X.fillna('Unknown')
    
    # 3. Label encode categorical columns (same as exp.py)
    for col in label_encoders.keys():
        if col in X.columns:
            try:
                X[col] = label_encoders[col].transform(X[col].astype(str))
            except Exception:
                # Handle unseen categories
                X[col] = 0
    
    # 4. Scale (same as exp.py)
    X_scaled = scaler.transform(X)
    
    # 5. Predict
    prediction_prob = model.predict(X_scaled)[0][0]
    
    # 6. Display Results
    st.write("---")
    if prediction_prob > 0.5:
        st.error(f"⚠️ **HIGH RISK of Late Delivery**")
        st.write(f"Probability: {prediction_prob:.2%}")
    else:
        st.success(f"✅ **LOW RISK** - Likely On-Time")
        st.write(f"Probability of delay: {prediction_prob:.2%}")

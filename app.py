import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Title and description
st.set_page_config(page_title="Zara Sales Predictor", layout="centered")
st.title("🛍️ Zara Sales Prediction App")
st.write("Enter product and store details to predict expected sales volume.")

# Load the trained model
@st.cache_resource
def load_model():
    with open('zara_sales_model.pkl', 'rb') as file:
        return pickle.load(file)

model = load_model()

# Sidebar inputs
st.sidebar.header("📋 Input Features")

product_category = st.sidebar.selectbox("Product Category", ["Tops", "Bottoms", "Dresses", "Accessories"])
price = st.sidebar.number_input("Price (in INR)", min_value=100.0, step=10.0)
store_location = st.sidebar.selectbox("Store Location", ["Mumbai", "Delhi", "Bangalore", "Hyderabad"])

# Predict button
if st.button("Predict Sales Volume"):
    # Create input DataFrame for prediction
    input_data = pd.DataFrame({
        'Product Category': [product_category],
        'Price': [price],
        'Store Location': [store_location]
    })

    # Perform prediction
    try:
        prediction = model.predict(input_data)[0]
        st.success(f"🔮 Estimated Sales Volume: **{int(prediction)} units**")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

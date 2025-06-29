
import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("zara_sales_model.pkl")

# App config
st.set_page_config(page_title="Zara Sales Predictor", layout="centered")
st.title("🛍️ Zara Sales Volume Predictor")
st.markdown("Predict expected **sales volume** based on product and store features.")

# Input fields (update categories as per your dataset)
product_category = st.selectbox("Product Category", ['Tops', 'Jeans', 'Dresses', 'Accessories'])
price = st.number_input("Product Price (₹)", min_value=100, max_value=10000, step=50)
store_location = st.selectbox("Store Location", ['Mumbai', 'Delhi', 'Bangalore', 'Kolkata'])
season = st.selectbox("Season", ['Spring', 'Summer', 'Autumn', 'Winter'])
promotion = st.selectbox("Promotion Running?", ['Yes', 'No'])

# Create input DataFrame for prediction
input_df = pd.DataFrame({
    'Product Category': [product_category],
    'Price': [price],
    'Store Location': [store_location],
    'Season': [season],
    'Promotion': [promotion]
})

# Make prediction
if st.button("Predict Sales Volume"):
    prediction = model.predict(input_df)
    st.success(f"📈 Predicted Sales Volume: **{int(prediction[0])} units**")

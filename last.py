import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Kaydedilmiş modeli yükle
model = joblib.load("catboost_house_rent.cbm")


st.title("🏠 House Rent Prediction App")

BHK = st.number_input("BHK (oda sayısı)", min_value=1, max_value=10, value=2)
Size = st.number_input("Size (sqft)", min_value=100, max_value=10000, value=1000)
Bathroom = st.number_input("Bathroom", min_value=1, max_value=10, value=1)
City = st.selectbox("City", ["Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai", "Kolkata"])
Area_Type = st.selectbox("Area Type", ["Super Area", "Carpet Area", "Build Area"])
Furnishing = st.selectbox("Furnishing Status", ["Furnished", "Semi-Furnished", "Unfurnished"])




input_data = pd.DataFrame([{
    "BHK": BHK,
    "Size": Size,
    "Bathroom": Bathroom,
    "City": City,
    "Area Type": Area_Type,
    "Furnishing Status": Furnishing,
    # Gerekirse diğer kolonları da ekle (Floor, Area Locality vs.)
}])



if st.button("Predict Rent"):
    pred = model.predict(input_data)
    pred_exp = np.expm1(pred)  # log-transform kullandıysan geri çevir
    st.success(f"Tahmini Kira: ₹{pred_exp[0]:,.0f}")

import streamlit as st
import pickle
import numpy as np

# Load the trained model
model_path = "cancer_risk_model.pkl"
with open(model_path, "rb") as file:
    model = pickle.load(file)

# Streamlit app
st.title("Cancer Risk Prediction")
st.write("This app predicts the cancer risk percentage based on input features.")

# User input for independent variables
st.sidebar.header("Input Features")
number = st.sidebar.selectbox("Number", list(range(0, 101)), index=10)
age = st.sidebar.selectbox("Age", list(range(0, 101)), index=30)
sex = st.sidebar.selectbox("Sex", ["M", "F"], index=1)
composition = st.sidebar.selectbox("Composition", ["solid", "predominantly solid", "other"], index=0)
echogenicity = st.sidebar.selectbox("Echogenicity", ["hyperechogenicity", "isoechogenicity", "hypoechogenicity", "other"], index=1)
margins = st.sidebar.selectbox("Margins", ["well defined", "spiculated", "other"], index=0)
calcifications = st.sidebar.selectbox("Calcifications", ["microcalcifications", "macrocalcifications", "none"], index=0)
tirads = st.sidebar.selectbox("TIRADS", ["3", "4a", "4b", "5"], index=1)
malignant_percentage = st.sidebar.selectbox("Malignant Percentage", [round(i * 0.01, 2) for i in range(0, 101)], index=50)

# Create input array
input_data = np.array([[
    number, age, sex, composition, echogenicity, margins, calcifications, tirads, malignant_percentage
]])

# Prediction
if st.button("Predict Cancer Risk"):
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Cancer Risk: {prediction:.2f}")

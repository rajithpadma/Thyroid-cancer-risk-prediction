import streamlit as st
import pickle
import numpy as np

# Load the trained model
model_path = "cancer_risk_model.pkl"
with open(model_path, "rb") as file:
    model = pickle.load(file)

# Encoding dictionaries (adjust these if needed to match your model's training)
sex_mapping = {"M": 0, "F": 1}
composition_mapping = {"solid": 0, "predominantly solid": 1, "other": 2}
echogenicity_mapping = {"hyperechogenicity": 0, "isoechogenicity": 1, "hypoechogenicity": 2, "other": 3}
margins_mapping = {"well defined": 0, "spiculated": 1, "other": 2}
calcifications_mapping = {"microcalcifications": 0, "macrocalcifications": 1, "none": 2}
tirads_mapping = {"3": 0, "4a": 1, "4b": 2, "5": 3}

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

# Encode user inputs
encoded_sex = sex_mapping[sex]
encoded_composition = composition_mapping[composition]
encoded_echogenicity = echogenicity_mapping[echogenicity]
encoded_margins = margins_mapping[margins]
encoded_calcifications = calcifications_mapping[calcifications]
encoded_tirads = tirads_mapping[tirads]

# Create the encoded input array
input_data = np.array([[
    number, age, encoded_sex, encoded_composition, encoded_echogenicity,
    encoded_margins, encoded_calcifications, encoded_tirads, malignant_percentage
]])

# Debugging (optional, remove or comment out later)
st.write("Input Data Shape:", input_data.shape)
st.write("Input Data:", input_data)

# Prediction
if st.button("Predict Cancer Risk"):
    try:
        prediction = model.predict(input_data)[0]
        st.success(f"Predicted Cancer Risk: {prediction:.2f}")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

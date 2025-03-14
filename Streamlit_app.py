import streamlit as st
import pickle
import pandas as pd

# Load the trained model
model_path = "cancer_risk_model.pkl"
with open(model_path, "rb") as file:
    model = pickle.load(file)

# Encoding dictionaries (adjust these to match your model's training)
sex_mapping = {"M": 0, "F": 1}
composition_mapping = {"solid": 0, "predominantly solid": 1, "other": 2}
echogenicity_mapping = {"hyperechogenicity": 0, "isoechogenicity": 1, "hypoechogenicity": 2, "other": 3}
margins_mapping = {"well defined": 0, "spiculated": 1, "other": 2}
calcifications_mapping = {"microcalcifications": 0, "macrocalcifications": 1, "none": 2}
tirads_mapping = {"3": 0, "4a": 1, "4b": 2, "5": 3}

# Expected column names
expected_columns = [
    "number", "age", "sex", "composition", "echogenicity", 
    "margins", "calcifications", "tirads", "Malignant_percentage"
]

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
encoded_inputs = {
    "number": number,
    "age": age,
    "sex": sex_mapping[sex],
    "composition": composition_mapping[composition],
    "echogenicity": echogenicity_mapping[echogenicity],
    "margins": margins_mapping[margins],
    "calcifications": calcifications_mapping[calcifications],
    "tirads": tirads_mapping[tirads],
    "Malignant_percentage": malignant_percentage,
}

# Convert to DataFrame
input_df = pd.DataFrame([encoded_inputs], columns=expected_columns)

# Debugging (optional, remove or comment out later)
st.write("Input DataFrame:")
st.write(input_df)

# Prediction
if st.button("Predict Cancer Risk"):
    try:
        prediction = model.predict(input_df)[0]
        st.success(f"Predicted Cancer Risk: {prediction:.2f}")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

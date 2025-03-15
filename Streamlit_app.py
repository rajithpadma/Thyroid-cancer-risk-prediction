import streamlit as st
import pickle
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the ML model
with open("cancer_risk_model.pkl", "rb") as file:
    ml_model = pickle.load(file)


# Encoding dictionaries for ML model
sex_mapping = {"M": 0, "F": 1}
composition_mapping = {"solid": 0, "predominantly solid": 1, "other": 2}
echogenicity_mapping = {"hyperechogenicity": 0, "isoechogenicity": 1, "hypoechogenicity": 2, "other": 3}
margins_mapping = {"well defined": 0, "spiculated": 1, "other": 2}
calcifications_mapping = {"microcalcifications": 0, "macrocalcifications": 1, "none": 2}
tirads_mapping = {"3": 0, "4a": 1, "4b": 2, "5": 3}

# Streamlit app
st.title("Thyroid Cancer and Nodule Prediction")

# Cancer risk prediction
st.sidebar.header("Cancer Risk Prediction Features")
age = st.sidebar.selectbox("Age", list(range(0, 101)), index=30)
sex = st.sidebar.selectbox("Sex", ["M", "F"], index=1)
composition = st.sidebar.selectbox("Composition", ["solid", "predominantly solid", "other"], index=0)
echogenicity = st.sidebar.selectbox("Echogenicity", ["hyperechogenicity", "isoechogenicity", "hypoechogenicity", "other"], index=1)
margins = st.sidebar.selectbox("Margins", ["well defined", "spiculated", "other"], index=0)
calcifications = st.sidebar.selectbox("Calcifications", ["microcalcifications", "macrocalcifications", "none"], index=0)
tirads = st.sidebar.selectbox("TIRADS", ["3", "4a", "4b", "5"], index=1)
malignant_percentage = st.sidebar.selectbox("Malignant Percentage", [round(i * 0.01, 2) for i in range(0, 101)], index=50)

encoded_inputs = {
    "age": age,
    "sex": sex_mapping[sex],
    "composition": composition_mapping[composition],
    "echogenicity": echogenicity_mapping[echogenicity],
    "margins": margins_mapping[margins],
    "calcifications": calcifications_mapping[calcifications],
    "tirads": tirads_mapping[tirads],
    "Malignant_percentage": malignant_percentage,
}
input_df = pd.DataFrame([encoded_inputs])

if st.sidebar.button("Predict Cancer Risk"):
    prediction = ml_model.predict(input_df)[0]
    st.success(f"Predicted Cancer Risk: {prediction:.2f}%")

# Load the trained CNN model
cnn_model = load_model("cnn_image_model.h5")

# Streamlit interface for cancer risk percentage prediction
st.header("Thyroid Cancer Risk Percentage Prediction")
uploaded_file = st.file_uploader("Upload Ultrasound Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load and preprocess the uploaded image
    img = image.load_img(uploaded_file, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict the cancer risk percentage using the CNN model
    risk_percentage = cnn_model.predict(img_array)[0][0]  # Extract the percentage value
    risk_percentage = round(risk_percentage * 100, 2)  # Convert to percentage and round off

    # Display the uploaded image and prediction
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.success(f"Predicted Thyroid Cancer Risk: {risk_percentage}%")

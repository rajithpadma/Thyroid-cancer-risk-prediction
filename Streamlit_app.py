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

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Constants
img_height, img_width = 224, 224

# Load the trained model
model = load_model("simplified_model.h5")

# Streamlit app interface
st.title("Thyroid Cancer Risk Prediction")
st.write("Upload an ultrasound image to predict the thyroid cancer risk percentage.")

uploaded_file = st.file_uploader("Choose an Ultrasound Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Preprocess the uploaded image
    img = load_img(uploaded_file, target_size=(img_height, img_width))
    img_array = img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict the risk percentage
    prediction = model.predict(img_array)
    risk_percentage = prediction[0][0] * 100  # Convert to percentage
    risk_percentage = round(risk_percentage, 2)  # Round to two decimal places

    # Display prediction
    st.success(f"Predicted Thyroid Cancer Risk: {risk_percentage}%")

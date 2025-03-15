import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pickle

# Constants
IMG_HEIGHT, IMG_WIDTH = 224, 224
ML_MODEL_PATH = "cancer_risk_model.pkl"
CNN_MODEL_PATH = "simplified_model.h5"

# Load models
@st.cache_resource
def load_models():
    try:
        # Load ML model
        with open(ML_MODEL_PATH, "rb") as file:
            ml_model = pickle.load(file)
        # Load CNN model
        cnn_model = load_model(CNN_MODEL_PATH)
        return ml_model, cnn_model
    except Exception as e:
        st.sidebar.error(f"Error loading models: {e}")
        return None, None

ml_model, cnn_model = load_models()

# Dictionaries for encoding categorical features
sex_mapping = {"M": 0, "F": 1}
composition_mapping = {"solid": 0, "predominantly solid": 1, "other": 2}
echogenicity_mapping = {"hyperechogenicity": 0, "isoechogenicity": 1, "hypoechogenicity": 2, "other": 3}
margins_mapping = {"well defined": 0, "spiculated": 1, "other": 2}
calcifications_mapping = {"microcalcifications": 0, "macrocalcifications": 1, "none": 2}
tirads_mapping = {"3": 0, "4a": 1, "4b": 2, "5": 3}

# Streamlit interface
st.title("Thyroid Cancer Risk Prediction")
st.sidebar.header("Choose a Prediction Mode")
mode = st.sidebar.radio("Prediction Type", ["ML Model (Tabular Data)", "CNN Model (Ultrasound Image)"])

if mode == "ML Model (Tabular Data)":
    st.header("Cancer Risk Prediction using Tabular Data")

    # Input fields
    st.sidebar.header("Features for ML Prediction")
    number = st.sidebar.slider("Number of Nodules", 0, 100, 10)
    age = st.sidebar.slider("Age", 0, 100, 30)
    sex = st.sidebar.selectbox("Sex", ["M", "F"], index=1)
    composition = st.sidebar.selectbox("Composition", ["solid", "predominantly solid", "other"], index=0)
    echogenicity = st.sidebar.selectbox("Echogenicity", ["hyperechogenicity", "isoechogenicity", "hypoechogenicity", "other"], index=1)
    margins = st.sidebar.selectbox("Margins", ["well defined", "spiculated", "other"], index=0)
    calcifications = st.sidebar.selectbox("Calcifications", ["microcalcifications", "macrocalcifications", "none"], index=0)
    tirads = st.sidebar.selectbox("TIRADS", ["3", "4a", "4b", "5"], index=1)
    malignant_percentage = st.sidebar.slider("Malignant Percentage", 0.0, 1.0, 0.5, step=0.01)

    # Prepare input for prediction
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
    input_df = pd.DataFrame([encoded_inputs])

    if st.sidebar.button("Predict Cancer Risk (ML Model)"):
        if ml_model:
            prediction = ml_model.predict(input_df)[0]
            st.success(f"Predicted Cancer Risk: {prediction:.2f}%")
        else:
            st.error("ML model not loaded. Check the model path or format.")

elif mode == "CNN Model (Ultrasound Image)":
    st.header("Cancer Risk Prediction using Ultrasound Image")
    st.write("Upload an ultrasound image to predict the thyroid cancer risk percentage.")

    uploaded_file = st.file_uploader("Choose an Ultrasound Image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        try:
            # Display uploaded image
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

            # Preprocess the uploaded image
            img = load_img(uploaded_file, target_size=(IMG_HEIGHT, IMG_WIDTH))
            img_array = img_to_array(img) / 255.0  # Normalize
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

            # Predict the risk percentage
            if cnn_model:
                prediction = cnn_model.predict(img_array)
                risk_percentage = prediction[0][0] * 100  # Convert to percentage
                risk_percentage = round(risk_percentage, 2)  # Round to two decimal places
                st.success(f"Predicted Thyroid Cancer Risk: {risk_percentage}%")
            else:
                st.error("CNN model not loaded. Check the model path or format.")
        except Exception as img_error:
            st.error(f"Error processing the uploaded image: {img_error}")
    else:
        st.info("Please upload an image to start the prediction.")

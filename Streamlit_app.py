import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# =========================
# CONFIGURATION
# =========================

IMG_HEIGHT = 224
IMG_WIDTH = 224

ML_MODEL_PATH = "cancer_risk_model.pkl"
CNN_MODEL_PATH = "simplified_model.h5"

# =========================
# LOAD MODELS
# =========================

@st.cache_resource
def load_models():
    ml_model = None
    cnn_model = None

    try:
        # Load ML model
        if os.path.exists(ML_MODEL_PATH):
            ml_model = joblib.load(ML_MODEL_PATH)
            st.sidebar.success("ML model loaded successfully.")
        else:
            st.sidebar.error(f"ML model not found: {ML_MODEL_PATH}")

        # Load CNN model
        if os.path.exists(CNN_MODEL_PATH):
            cnn_model = load_model(CNN_MODEL_PATH)
            st.sidebar.success("CNN model loaded successfully.")
        else:
            st.sidebar.error(...)

    except Exception as e:
        st.sidebar.error(...)

    return ml_model, cnn_model

# Load models
ml_model, cnn_model = load_models()

# =========================
# CATEGORY ENCODINGS
# =========================

sex_mapping = {
    "M": 0,
    "F": 1
}

composition_mapping = {
    "solid": 0,
    "predominantly solid": 1,
    "other": 2
}

echogenicity_mapping = {
    "hyperechogenicity": 0,
    "isoechogenicity": 1,
    "hypoechogenicity": 2,
    "other": 3
}

margins_mapping = {
    "well defined": 0,
    "spiculated": 1,
    "other": 2
}

calcifications_mapping = {
    "microcalcifications": 0,
    "macrocalcifications": 1,
    "none": 2
}

tirads_mapping = {
    "3": 0,
    "4a": 1,
    "4b": 2,
    "5": 3
}

# =========================
# UI
# =========================

st.title("Thyroid Cancer Risk Prediction System")

st.sidebar.header("Prediction Mode")

mode = st.sidebar.radio(
    "Choose Prediction Type",
    [
        "ML Model (Tabular Data)",
        "CNN Model (Ultrasound Image)"
    ]
)

# =========================================================
# ML MODEL SECTION
# =========================================================

if mode == "ML Model (Tabular Data)":

    st.header("Cancer Risk Prediction using Tabular Data")

    st.sidebar.header("Patient Features")

    number = st.sidebar.slider("Number of Nodules", 0, 100, 10)

    age = st.sidebar.slider("Age", 1, 100, 30)

    sex = st.sidebar.selectbox(
        "Sex",
        ["M", "F"]
    )

    composition = st.sidebar.selectbox(
        "Composition",
        ["solid", "predominantly solid", "other"]
    )

    echogenicity = st.sidebar.selectbox(
        "Echogenicity",
        [
            "hyperechogenicity",
            "isoechogenicity",
            "hypoechogenicity",
            "other"
        ]
    )

    margins = st.sidebar.selectbox(
        "Margins",
        [
            "well defined",
            "spiculated",
            "other"
        ]
    )

    calcifications = st.sidebar.selectbox(
        "Calcifications",
        [
            "microcalcifications",
            "macrocalcifications",
            "none"
        ]
    )

    tirads = st.sidebar.selectbox(
        "TIRADS",
        ["3", "4a", "4b", "5"]
    )

    malignant_percentage = st.sidebar.slider(
        "Malignant Percentage",
        0.0,
        1.0,
        0.5,
        step=0.01
    )

    # Create dataframe

    encoded_inputs = {
        "number": number,
        "age": age,
        "sex": sex_mapping[sex],
        "composition": composition_mapping[composition],
        "echogenicity": echogenicity_mapping[echogenicity],
        "margins": margins_mapping[margins],
        "calcifications": calcifications_mapping[calcifications],
        "tirads": tirads_mapping[tirads],
        "Malignant_percentage": malignant_percentage
    }

    input_df = pd.DataFrame([encoded_inputs])

    if st.sidebar.button("Predict Cancer Risk"):

        if ml_model is not None:

            try:
                prediction = ml_model.predict(input_df)[0]

                st.success(
                    f"Predicted Cancer Risk: {round(prediction, 2)}%"
                )

            except Exception as pred_error:
                st.error(f"Prediction Error:\n{pred_error}")

        else:
            st.error("ML model failed to load.")

# =========================================================
# CNN MODEL SECTION
# =========================================================

elif mode == "CNN Model (Ultrasound Image)":

    st.header("Cancer Risk Prediction using Ultrasound Images")

    uploaded_file = st.file_uploader(
        "Upload Ultrasound Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        try:
            st.image(
                uploaded_file,
                caption="Uploaded Ultrasound Image",
                width=400
            )

            # Image preprocessing

            img = load_img(
                uploaded_file,
                target_size=(IMG_HEIGHT, IMG_WIDTH)
            )

            img_array = img_to_array(img)

            img_array = img_array / 255.0

            img_array = np.expand_dims(img_array, axis=0)

            if cnn_model is not None:

                prediction = cnn_model.predict(img_array)

                risk_percentage = float(prediction[0][0]) * 100

                st.success(
                    f"Predicted Thyroid Cancer Risk: {round(risk_percentage, 2)}%"
                )

            else:
                st.error("CNN model failed to load.")

        except Exception as img_error:
            st.error(f"Image Processing Error:\n{img_error}")

    else:
        st.info("Please upload an ultrasound image.")

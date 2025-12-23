Artificial Intelligence–Powered CNN for Predicting Cancer Risk from Thyroid Nodule Ultrasound Images
Project Overview

This project presents an Artificial Intelligence-based hybrid cancer risk prediction system designed to evaluate thyroid nodule ultrasound images and clinical health parameters to estimate malignancy risk. Traditional thyroid cancer diagnosis often relies heavily on radiologist expertise and manual interpretation, which can result in subjectivity, inconsistency, and variability in diagnostic accuracy.

To address these challenges, this system integrates MobileNetV2-based Convolutional Neural Networks (CNN) for image analysis with a Decision Tree Machine Learning model for structured clinical data assessment. The system supports dual prediction modes—image-based cancer risk prediction and manual clinical input-based prediction—and is implemented with an intuitive Streamlit interface suitable for real clinical environments.

Problem Statement

Diagnosing malignant thyroid nodules is complex due to:

Dependence on clinician interpretation

Dataset imbalance affecting prediction accuracy

Limited precision in traditional feature extraction methods

Lack of scalable, real-time diagnostic support systems

Healthcare professionals therefore require an automated and intelligent system that improves diagnostic confidence and enhances early decision-making.

Abstract

This project develops a hybrid AI model that combines CNN-based thyroid ultrasound image analysis with Decision Tree-based structured data evaluation. Using MobileNetV2 for efficient feature extraction, the system achieves highly accurate risk prediction (R² ≈ 98%). Dual functionality allows prediction through either ultrasound image upload or manually entered clinical parameters, with results displayed via a user-friendly Streamlit interface.

Key Functionalities
1. Image-Based Prediction

Users upload a thyroid ultrasound image

The CNN model analyzes the image

The system predicts cancer risk percentage

2. Manual Input-Based Prediction

Users manually input clinical health parameters
(e.g., thyroid values, patient characteristics, nodule attributes, etc.)

The Decision Tree model evaluates the data

The system predicts cancer risk percentage

Useful when image data is not available

System Advantages

Lightweight MobileNetV2 ensures faster inference

Suitable for real-time and edge deployment

Practical Streamlit interface

Dual prediction flexibility

Enhanced diagnostic accuracy and confidence

Technology Stack

Programming Language: Python
Deep Learning Framework: TensorFlow / Keras
CNN Architecture: MobileNetV2
Machine Learning Model: Decision Tree
Interface: Streamlit
Dataset: Thyroid Ultrasound Image Dataset

System Workflow

The user either uploads an ultrasound image or inputs clinical parameters

CNN model processes image data, while the Decision Tree processes manual inputs

Cancer risk percentage is calculated

Results are displayed with interpretation

Supports clinical insights and risk understanding

How to Run the Project
Step 1: Clone the Repository
git clone <repository_link>
cd <project_folder>

Step 2: Install Dependencies
pip install -r requirements.txt

Step 3: Run the Application
streamlit run app.py

Step 4: Using the Application

Once the application opens in the browser:

Option 1 — Upload Ultrasound Image

Upload a thyroid ultrasound image

The system analyzes the image

Cancer risk percentage is displayed

Option 2 — Manual Entry Mode

Enter the required clinical parameters manually

Submit the form

Cancer risk percentage is displayed

Results and Discussion

Achieved R² ≈ 98% accuracy

Handles dataset imbalance effectively

Provides fast and reliable predictions

Demonstrates practical AI-assisted healthcare capability

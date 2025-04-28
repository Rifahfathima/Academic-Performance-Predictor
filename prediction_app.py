import streamlit as st
import pandas as pd
import joblib
import os

# Step 1: Load the trained model
model_path = 'saved_models/student_performance_model.pkl'

if not os.path.exists(model_path):
    st.error(f"❌ Model file not found at {model_path}. Please train the model first.")
else:
    model = joblib.load(model_path)
    st.success("✅ Model loaded successfully!")

    # Step 2: Create an input form for the user to provide data
    st.title("Student Performance Prediction")
    st.write("Enter the following details to predict final grade:")

    # User inputs for each feature
    study_hours = st.number_input("Study Hours", min_value=0.0, max_value=24.0, value=6.0)
    attendance = st.slider("Attendance (%)", min_value=0, max_value=100, value=90)
    assignments = st.slider("Assignments Completed", min_value=0, max_value=10, value=8)
    past_grade = st.slider("Past Grade (%)", min_value=0, max_value=100, value=75)
    participation = st.slider("Class Participation (1-10)", min_value=1, max_value=10, value=7)
    sleep_hours = st.number_input("Sleep Hours", min_value=0, max_value=24, value=7)

    # Button to make the prediction
    if st.button("Predict Final Grade"):
        # Prepare the data in the format the model expects
        input_data = pd.DataFrame({
            'study_hours': [study_hours],
            'attendance': [attendance],
            'assignments': [assignments],
            'past_grade': [past_grade],
            'participation': [participation],
            'sleep_hours': [sleep_hours]
        })

        # Step 3: Make prediction
        prediction = model.predict(input_data)

        # Step 4: Display the result
        st.write(f"🎯 Predicted Final Grade: {prediction[0]:.2f}")

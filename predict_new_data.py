# # scripts/predict_new_data.py

# import pandas as pd
# import joblib

# # Load the trained model
# model = joblib.load('saved_models/student_performance_model.pkl')

# # New data for prediction (Replace with your own values)
# new_data = {
#     'study_hours': [6.5, 8.2], 
#     'attendance': [88, 92], 
#     'assignments': [9, 10], 
#     'past_grade': [75, 85],
#     'participation': [8, 9], 
#     'sleep_hours': [7, 8]
# }

# # Convert the new data into a DataFrame
# new_data_df = pd.DataFrame(new_data)

# # Make predictions
# predictions = model.predict(new_data_df)

# # Output the predictions
# for i, prediction in enumerate(predictions, 1):
#     print(f"Prediction {i}: The predicted final grade for student {i} is {prediction:.2f}")


# scripts/predict_new_data.py

import pandas as pd
import joblib
import os

# Step 1: Load the trained model
model_path = 'saved_models/student_performance_model.pkl'

if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Model file not found at {model_path}. Please train the model first.")

model = joblib.load(model_path)
print("✅ Model loaded successfully!")

# Step 2: Prepare new data for prediction
# Example: New students' data
new_data = pd.DataFrame({
    'study_hours': [6.5, 8.0, 4.0],
    'attendance': [90, 95, 75],
    'assignments': [8, 10, 5],
    'past_grade': [70, 85, 60],
    'participation': [7, 9, 4],
    'sleep_hours': [7, 8, 6]
})

print("\n🆕 New Data for Prediction:")
print(new_data)

# Step 3: Make predictions
predictions = model.predict(new_data)

# Step 4: Display the results
print("\n🎯 Predicted Final Grades:")
for idx, grade in enumerate(predictions):
    print(f"Student {idx+1}: Predicted Final Grade = {grade:.2f}")

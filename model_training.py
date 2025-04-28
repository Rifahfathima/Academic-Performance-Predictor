# # scripts/model_training.py

# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# import joblib
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Load dataset
# data = pd.read_csv('data/student_data.csv')

# # Clean up column names to remove extra spaces
# data.columns = data.columns.str.strip()

# # Check the column names to make sure everything is correct
# print("Columns in the dataset:", data.columns)

# # Features and Target Variable
# X = data[['study_hours', 'attendance', 'assignments', 'past_grade', 'participation', 'sleep_hours']]
# y = data['final_grade']

# # Split the dataset into training and testing sets (80% train, 20% test)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Build and Train Model
# model = LinearRegression()
# model.fit(X_train, y_train)

# # Make Predictions on the Test Data
# y_pred = model.predict(X_test)

# # Model Evaluation
# print("\nModel Evaluation:")
# print(f"Mean Absolute Error (MAE): {mean_absolute_error(y_test, y_pred):.2f}")
# print(f"Mean Squared Error (MSE): {mean_squared_error(y_test, y_pred):.2f}")
# print(f"Root Mean Squared Error (RMSE): {(mean_squared_error(y_test, y_pred))**0.5:.2f}")
# print(f"R2 Score: {r2_score(y_test, y_pred):.2f}")

# # Save the trained model to a file for future use
# joblib.dump(model, 'saved_models/student_performance_model.pkl')
# print("\nModel saved successfully!")

# # Visualizations

# # 1. Correlation Heatmap
# plt.figure(figsize=(10, 8))
# correlation = data.corr()  # Calculate correlations
# sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
# plt.title("Correlation Heatmap of Features")
# plt.show()

# # 2. Scatter Plot for True vs Predicted Final Grades
# plt.figure(figsize=(8, 6))
# plt.scatter(y_test, y_pred, color='blue')
# plt.plot([y.min(), y.max()], [y.min(), y.max()], '--r', color='red')
# plt.title("True vs Predicted Final Grades")
# plt.xlabel("True Final Grades")
# plt.ylabel("Predicted Final Grades")
# plt.show()







# scripts/model_training.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

# Step 1: Load dataset
data = pd.read_csv('data/student_data.csv')

# Step 2: Clean up column names to remove extra spaces
data.columns = data.columns.str.strip()

# Confirm columns
print("✅ Columns in the dataset:", list(data.columns))

# Step 3: Define Features and Target Variable
feature_cols = ['study_hours', 'attendance', 'assignments', 'past_grade', 'participation', 'sleep_hours']
X = data[feature_cols]
y = data['final_grade']

# Step 4: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Build and Train the Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Make Predictions on the Test Data
y_pred = model.predict(X_test)

# Step 7: Model Evaluation
print("\n📊 Model Evaluation Metrics:")
print(f"Mean Absolute Error (MAE): {mean_absolute_error(y_test, y_pred):.2f}")
print(f"Mean Squared Error (MSE): {mean_squared_error(y_test, y_pred):.2f}")
print(f"Root Mean Squared Error (RMSE): {(mean_squared_error(y_test, y_pred))**0.5:.2f}")
print(f"R² Score: {r2_score(y_test, y_pred):.2f}")

# Step 8: Save the trained model to a file
model_save_path = 'saved_models/student_performance_model.pkl'
# Create folder if it doesn't exist
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

joblib.dump(model, model_save_path)
print(f"\n✅ Model saved successfully at '{model_save_path}'!")

# Step 9: Visualizations

## 9.1 Correlation Heatmap
plt.figure(figsize=(10, 8))
correlation = data.corr()
sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Student Features")
plt.tight_layout()
plt.show()

## 9.2 True vs Predicted Final Grades Scatter Plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], linestyle='--', color='red')
plt.title("True vs Predicted Final Grades")
plt.xlabel("True Final Grades")
plt.ylabel("Predicted Final Grades")
plt.grid(True)
plt.tight_layout()
plt.show()


#      python scripts/model_training.py


#      python scripts/predict_new_data.py

#      python scripts/data_visualization.py


#      streamlit run scripts/prediction_app.py
# import streamlit as st
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# import pickle

# # Load trained model
# model = pickle.load(open('saved_models/student_performance_model.pkl', 'rb'))

# # Title
# st.title("🎓 Student Performance Predictor and Visualizer")

# # Sidebar for Navigation
# option = st.sidebar.selectbox("Choose an option", ("Home", "Visualizations", "Predict Final Grade"))

# # Home Page
# if option == "Home":
#     st.write("""
#         Welcome to the **Student Performance App**!
        
#         Here you can visualize past student data and predict final grades based on study patterns 📚
#     """)

# # Visualization Page
# elif option == "Visualizations":
#     data = pd.read_csv('data/student_data.csv')
#     st.subheader("Data Visualizations")
    
#     # Histogram
#     st.write("### Final Grade Distribution")
#     fig1, ax1 = plt.subplots()
#     sns.histplot(data['final_grade'], kde=True, ax=ax1)
#     st.pyplot(fig1)
    
#     # Correlation Heatmap
#     st.write("### Correlation Heatmap")
#     fig2, ax2 = plt.subplots()
#     sns.heatmap(data.corr(), annot=True, cmap='coolwarm', ax=ax2)
#     st.pyplot(fig2)

# # Prediction Page
# elif option == "Predict Final Grade":
#     st.subheader("Enter Details to Predict Grade")
    
#     study_hours = st.number_input("Study Hours", min_value=0.0, max_value=15.0, step=0.1)
#     attendance = st.number_input("Attendance (%)", min_value=0, max_value=100, step=1)
#     assignments = st.number_input("Assignments Completed", min_value=0, max_value=10, step=1)
#     past_grade = st.number_input("Past Grade (%)", min_value=0, max_value=100, step=1)
#     participation = st.number_input("Participation Level", min_value=0, max_value=10, step=1)
#     sleep_hours = st.number_input("Average Sleep Hours", min_value=0, max_value=12, step=1)
    
#     if st.button("Predict"):
#         input_data = [[study_hours, attendance, assignments, past_grade, participation, sleep_hours]]
#         prediction = model.predict(input_data)
#         st.success(f"🎯 Predicted Final Grade: {prediction[0]:.2f}%")


import streamlit as st
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

# Load trained model
try:
    model = pickle.load(open('saved_models/student_performance_model.pkl', 'rb'))
    st.write("Model loaded successfully!")
    st.write(f"Loaded model type: {type(model)}")  # This will output the model type
except Exception as e:
    st.write(f"Error loading model: {str(e)}")

# Title
st.title("🎓 Student Performance Predictor and Visualizer")

# Sidebar for Navigation
option = st.sidebar.selectbox("Choose an option", ("Home", "Visualizations", "Predict Final Grade"))

# Home Page
if option == "Home":
    st.write("""
        Welcome to the **Student Performance App**!
        
        Here you can visualize past student data and predict final grades based on study patterns 📚
    """)

# Visualization Page
elif option == "Visualizations":
    data = pd.read_csv('data/student_data.csv')
    st.subheader("Data Visualizations")
    
    # Histogram
    st.write("### Final Grade Distribution")
    fig1, ax1 = plt.subplots()
    sns.histplot(data['final_grade'], kde=True, ax=ax1)
    st.pyplot(fig1)
    
    # Correlation Heatmap
    st.write("### Correlation Heatmap")
    fig2, ax2 = plt.subplots()
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', ax=ax2)
    st.pyplot(fig2)

# Prediction Page
elif option == "Predict Final Grade":
    st.subheader("Enter Details to Predict Grade")
    
    study_hours = st.number_input("Study Hours", min_value=0.0, max_value=15.0, step=0.1)
    attendance = st.number_input("Attendance (%)", min_value=0, max_value=100, step=1)
    assignments = st.number_input("Assignments Completed", min_value=0, max_value=10, step=1)
    past_grade = st.number_input("Past Grade (%)", min_value=0, max_value=100, step=1)
    participation = st.number_input("Participation Level", min_value=0, max_value=10, step=1)
    sleep_hours = st.number_input("Average Sleep Hours", min_value=0, max_value=12, step=1)
    
    if st.button("Predict"):
        input_data = [[study_hours, attendance, assignments, past_grade, participation, sleep_hours]]
        prediction = model.predict(input_data)  # This will call the predict() method from scikit-learn model
        st.success(f"🎯 Predicted Final Grade: {prediction[0]:.2f}%")

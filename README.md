# Student Academic Performance Predictor

This project predicts final student grades based on various academic and behavioral factors using regression models.
Built during the AI/ML Workshop to demonstrate basic machine learning pipelines and Streamlit apps!

📂 Project Structure
bash
Copy
Edit
StudentGradeProject/
├── data/               # Folder containing the student dataset
│   └── student_data.csv
├── saved_models/       # Trained model (Pickle file)
│   └── student_performance_model.pkl
├── scripts/            # Source code scripts
│   ├── data_visualization.py
│   ├── main.py
│   ├── model_training.py
│   ├── predict_new_data.py
│   ├── prediction_app.py
│   └── streamlit_app.py
├── stu_venv/           # Virtual environment (do not push to GitHub)
├── README.md           # Project description
├── requirements.txt    # Python dependencies
⚙️ Installation
Clone or download this project folder.

(Optional) Create a virtual environment.

Install required libraries:

bash
Copy
Edit
pip install -r requirements.txt
🚀 How to Train the Model
Make sure you have the student_data.csv file inside the data/ folder.

Then run:

bash
Copy
Edit
python scripts/model_training.py
✅ This will:

Train a Linear Regression model.

Save the model as saved_models/student_performance_model.pkl.

Optionally show evaluation metrics.

🤖 How to Predict Student Grades
If you want to predict new student grades using the saved model:

bash
Copy
Edit
python scripts/predict_new_data.py --input_data your_input.csv
✅ Example Output:

yaml
Copy
Edit
Predicted Grade: 85.3
🖥️ How to Use the Streamlit Web App
Launch the interactive web app with:

bash
Copy
Edit
streamlit run scripts/streamlit_app.py
✅ This will:

Open a local browser window.

Allow user to input student features manually.

Predict and display final grade instantly.

📋 Features
Inputs like Study Hours, Attendance, Past Grades, Assignments, Participation, Sleep Hours

Uses Linear Regression algorithm

Interactive Streamlit UI for easy prediction

📢 Notes
Dataset should be properly cleaned before training.

Virtual environment folder (stu_venv/) should be ignored when uploading to GitHub.

Model is saved automatically after training.

Web app is built with simple and clean Streamlit interface.

🧠 Credits
Scikit-learn

Pandas

Streamlit

Matplotlib



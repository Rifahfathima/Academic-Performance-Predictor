# scripts/data_visualization.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv('data/student_data.csv')

# Clean up column names to remove extra spaces
data.columns = data.columns.str.strip()

# Display first few rows of the dataset to ensure it loaded correctly
print("First few rows of the dataset:\n", data.head())

# Visualize the distribution of the final grades
plt.figure(figsize=(8, 6))
sns.histplot(data['final_grade'], kde=True, color='skyblue')
plt.title("Distribution of Final Grades")
plt.xlabel("Final Grade")
plt.ylabel("Frequency")
plt.show()

# Correlation heatmap to identify relationships between features
plt.figure(figsize=(10, 8))
correlation = data.corr()  # Calculate correlations
sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Features")
plt.show()

# Pairplot for a more detailed look at feature relationships
sns.pairplot(data)
plt.suptitle("Pairplot of Features", y=1.02)
plt.show()

# Boxplot for final grade distribution across different study hours
plt.figure(figsize=(8, 6))
sns.boxplot(x='study_hours', y='final_grade', data=data, palette="Set2")
plt.title("Boxplot of Final Grades by Study Hours")
plt.xlabel("Study Hours")
plt.ylabel("Final Grade")
plt.show()

# Scatter plot between Study Hours and Final Grades
plt.figure(figsize=(8, 6))
sns.scatterplot(x='study_hours', y='final_grade', data=data, color='orange')
plt.title("Study Hours vs Final Grade")
plt.xlabel("Study Hours")
plt.ylabel("Final Grade")
plt.show()

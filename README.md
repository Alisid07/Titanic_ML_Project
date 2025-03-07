Titanic Survival Prediction - Machine Learning Project

📌 Project Overview

This project applies Machine Learning (ML) techniques to predict passenger survival on the Titanic dataset. Using advanced data analysis, feature engineering, and model optimization, we improve prediction accuracy and visualize key insights.

📊 Dataset Information

The dataset is sourced from Data Science Dojo and includes:

Passenger details (Age, Sex, Fare, Pclass, SibSp, Parch, etc.)

Survival status (0 = No, 1 = Yes)

Categorical & Numerical features used for model training.

⚡ Key Features & Methods Used

✅ Data Cleaning & Preprocessing: Handling missing values, encoding categorical features.
✅ Exploratory Data Analysis (EDA): Detailed visualizations of survival rates, class distribution, and correlation heatmaps.
✅ Feature Engineering: Transforming raw data into meaningful insights.
✅ Machine Learning Model: Using Random Forest Classifier with hyperparameter tuning for optimal performance.
✅ Model Evaluation: Accuracy, classification report, confusion matrix, and feature importance analysis.

🛠️ Installation & Requirements

To run this project, install the necessary Python packages:

pip install pandas numpy matplotlib seaborn scikit-learn

🚀 Code Implementation

1️⃣ Load Dataset & Display Overview

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load Dataset
df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
print("\nFirst few rows of the dataset:")
print(df.head())

2️⃣ Exploratory Data Analysis (EDA)

Age Distribution

plt.figure(figsize=(8,5))
sns.histplot(df['Age'].dropna(), bins=30, kde=True, color='blue')
plt.title("Age Distribution of Passengers")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

Survival Rate by Gender

plt.figure(figsize=(6,4))
sns.barplot(x=df['Sex'], y=df['Survived'], estimator=np.mean, palette='coolwarm')
plt.title("Survival Rate by Gender")
plt.xlabel("Gender")
plt.ylabel("Survival Rate")
plt.show()

Correlation Heatmap

plt.figure(figsize=(10,5))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Feature Correlation Heatmap")
plt.show()

3️⃣ Data Preprocessing

# Fill missing values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop(columns=['Cabin', 'Name', 'Ticket'], inplace=True)

# Convert categorical columns to numeric
le = LabelEncoder()
for col in ['Sex', 'Embarked']:
    df[col] = le.fit_transform(df[col])

4️⃣ Machine Learning Model

# Feature Selection
X = df.drop(columns=['Survived'])
y = df['Survived']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model Training & Hyperparameter Tuning
rf = RandomForestClassifier()
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

5️⃣ Model Evaluation

# Predictions & Metrics
y_pred = best_model.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

Confusion Matrix

plt.figure(figsize=(6,5))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

Feature Importance

feature_importances = best_model.feature_importances_
features = X.columns
plt.figure(figsize=(10,5))
sns.barplot(x=feature_importances, y=features, palette='coolwarm')
plt.title("Feature Importance in Random Forest Model")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.show()

📈 Results & Conclusion

🔹 Accuracy Achieved: Over 80% using Random Forest Classifier.🔹 Feature Insights: Passenger class, fare, and gender strongly influence survival.🔹 EDA Visuals: Enhanced plots for better understanding of dataset trends.

This project demonstrates advanced ML techniques for predicting Titanic survival outcomes with robust analysis and hyperparameter tuning.

⭐ Contribute

Feel free to fork, improve, and contribute to this project!

🔗 GitHub Repository: Titanic_ML_Project

💡 If you like this project, star ⭐ the repo! 🚀

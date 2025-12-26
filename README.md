🧠 Autism Spectrum Disorder (ASD) Prediction using Machine Learning
📌 Project Overview

This project focuses on the analysis and prediction of Autism Spectrum Disorder (ASD) using machine learning techniques.
It uses data visualization and classification models to identify patterns and predict whether an individual is likely to have ASD.

The project applies Random Forest and K-Nearest Neighbors (KNN) algorithms and compares their performance to determine the most effective model.

🎯 Objectives

To analyze ASD screening data using Exploratory Data Analysis (EDA)

To visualize important patterns related to age, gender, and autism diagnosis

To preprocess and encode categorical data

To build and evaluate Random Forest and KNN classification models

To compare model accuracy and performance

📂 Dataset Description

The dataset contains demographic and screening-related features.

Target variable: Class/ASD

0 → No Autism

1 → Autism

Key Features:

Age

Gender

Ethnicity

Jaundice

Country of residence

Family history of autism

Screening test results

📊 Exploratory Data Analysis (EDA)

The following visualizations were used:

Count Plot: Autism vs Non-Autism distribution

Gender Distribution Plot

Autism distribution based on gender

Histogram with KDE: Age distribution

Mean and Median comparison to identify skewness

📌 Observation:
The age distribution is right-skewed, with most individuals belonging to younger age groups. ASD cases are more prevalent among males.

⚙️ Data Preprocessing

Missing values were checked

Categorical features were converted into numerical format using Label Encoding

Dataset was split into training (80%) and testing (20%)

🤖 Machine Learning Models Used
1️⃣ Random Forest Classifier

Ensemble learning method using multiple decision trees

Handles non-linearity and feature interactions effectively

Achieved higher accuracy compared to KNN

2️⃣ K-Nearest Neighbors (KNN)

Distance-based classification algorithm

Simple and intuitive

Performance depends on choice of k

📈 Model Evaluation Metrics

Accuracy Score

Confusion Matrix

Classification Report (Precision, Recall, F1-score)

📌 Result:
Random Forest outperformed KNN in terms of accuracy and overall classification performance.

🏆 Conclusion

The dataset shows clear patterns related to ASD

Visualization helped in understanding data distribution

Random Forest proved to be the best performing model

Machine learning can support early ASD screening and diagnosis

🛠️ Technologies Used

Python

Pandas

NumPy

Matplotlib

Seaborn

Scikit-learn

📁 Project Structure
Autism-Prediction-ML/
│
├── dataset/
│   └── train.csv
│
├── notebooks/
│   └── ASD_Analysis.ipynb
│
├── README.md

📌 Future Enhancements

Apply deep learning models

Use feature selection techniques

Deploy model using Flask or Streamlit

Handle data imbalance using SMOTE (optional)

👨‍🎓 Author

Prince Kumar

📜 License

This project is developed for educational and academic purposes only.

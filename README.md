# 🎓 Student Performance Prediction

## 📌 Project Overview

Student Performance Prediction is a Machine Learning project that predicts a student's final examination score based on academic and personal factors.

The project uses **Random Forest Regression** and provides a simple **Streamlit web application** for making predictions.

## 🎯 Objective

The main objective of this project is to use Machine Learning to estimate a student's final score using:

* Study Hours
* Attendance
* Previous Exam Score
* Assignment Score
* Sleep Hours

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Joblib
* Streamlit

## 🤖 Machine Learning Model

**Algorithm:** Random Forest Regression

The dataset contains **500 synthetically generated student records**.

The dataset is divided into:

* 80% Training Data
* 20% Testing Data

## 📊 Model Performance

The model achieved the following results on the test dataset:

| Metric   | Result |
| -------- | -----: |
| MAE      |   3.13 |
| MSE      |  15.12 |
| R² Score | 0.8335 |

The R² score indicates how well the model explains variation in the target final score on the test data.

## 📂 Project Structure

```text
Student_Performance_Prediction/
│
├── venv/
├── student_data.csv
├── generate_dataset.py
├── train_model.py
├── student_performance_model.pkl
├── app.py
├── requirements.txt
└── README.md
```

## ⚙️ How to Run the Project

### Step 1: Create Virtual Environment

```bash
python -m venv venv
```

### Step 2: Activate Virtual Environment

Windows:

```bash
venv\Scripts\activate
```

### Step 3: Install Required Libraries

```bash
pip install -r requirements.txt
```

### Step 4: Train the Model

```bash
python train_model.py
```

This creates:

```text
student_performance_model.pkl
```

### Step 5: Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open in the web browser.

## 🔮 Prediction

The user enters:

* Study Hours per Day
* Attendance
* Previous Exam Score
* Assignment Score
* Sleep Hours

The trained model then predicts the student's final examination score.

## ⚠️ Disclaimer

This project is developed for educational and demonstration purposes.

The dataset used in this project is synthetically generated and should not be treated as real student performance data or as a reliable tool for making actual academic decisions.

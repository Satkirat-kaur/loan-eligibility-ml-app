# Loan Eligibility Prediction System

## Project Overview

This project predicts whether a loan application will be approved based on applicant financial and demographic details using Machine Learning.

The original Jupyter Notebook model was modularized into a production-ready Python project and deployed as an interactive web application using Streamlit.

---

## Objectives

* Convert notebook-based ML code into modular Python scripts
* Improve code readability, reusability, and maintainability
* Implement logging and error handling
* Deploy an interactive ML application using Streamlit
* Publish the project on GitHub

---

## Machine Learning Approach

### Dataset

The dataset contains loan applicant details such as:

* Applicant Income
* Coapplicant Income
* Loan Amount
* Credit History
* Property Area
* Education, Gender, etc.

### Target Variable

* **Loan_Approved**

  * `1` → Approved
  * `0` → Not Approved

---

## Data Processing Steps

* Handling missing values (mode & median imputation)
* Converting categorical variables
* One-hot encoding using `pd.get_dummies()`
* Feature scaling using **MinMaxScaler**
* Train-test split with stratification

---

## Model Used

* **Logistic Regression**

### Model Performance

* **Accuracy:** 85.37%
* Strong performance in predicting approved loans
* Very low false negative rate (important in financial decisions)

---

## Key Features of the Application

### Modular Code Structure

* Data preprocessing
* Feature engineering
* Model training
* Evaluation
* Utility functions

### Logging & Error Handling

* Logs stored in `/logs/app.log`
* Helps debug and monitor execution

### Interactive Streamlit App

* User-friendly interface
* Real-time loan prediction
* Displays:

  * Approval status
  * Probability score
  * Human-readable explanation of prediction

---

## Explanation Feature

The app includes a rule-based explanation system that highlights:

* High loan amount
* Low income
* Weak credit history
* Other financial indicators

This improves transparency and user understanding.

---

## Project Structure

```
loan-eligibility-ml-app/
│
├── data/
│   └── credit.csv
│
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   └── utils.py
│
├── models/
│   ├── loan_model.pkl
│   ├── scaler.pkl
│   └── feature_columns.pkl
│
├── logs/
│   └── app.log
│
├── app.py
├── main.py
├── requirements.txt
└── README.md
```

---

## How to Run the Project

### 1. Install Dependencies

```
pip install -r requirements.txt
```

### 2️. Train the Model

```
python main.py
```

### 3️. Run the Streamlit App

```
streamlit run app.py
```

---



## Sample Results

* High loan amount → Lower approval probability
* Strong credit history → Higher approval probability
* Balanced income → Increased chances of approval

---

## Future Improvements

* Use advanced models (XGBoost, Neural Networks)
* Add SHAP for explainability
* Improve UI/UX design
* Integrate real-time API

---

##  Author

**Satkirat Kaur**
Algonquin College
Business Intelligence System Infrastructure

---

##  GitHub Repository

https://github.com/Satkirat-kaur/loan-eligibility-ml-app

---

## Deployed Streamlit App

https://loan-eligibility-ml-app-8aamk8fekpz5bwxcuz8gke.streamlit.app/
---

##  Notes

This project demonstrates an end-to-end ML pipeline:

* From notebook → modular code → deployed application

It reflects real-world industry practices in Machine Learning deployment.

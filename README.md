# discoverchurnbymb
This repository showcases a data science project focusing on churn prediction. It features a Machine Learning model and an interactive Streamlit app for real-time predictions and visualizations. Specifically, it is based on a dataset containing information about male and female bank clients from Spain, Germany, and France. The focus is the prediction of churn in dependence on specific features, namely age, gender, Credit Score, tenure, balance, number of products, the possession of a credit card, activity, salary, and geography.

## Installazione

1. **Clone the repository:**
   ```bash
   git clone https://github.com/martinabavetta7/discoverchurnbymb.git
   ```

2. **Create a conda environment:**
   ```bash
   conda env create -f environment.yml
   ```

3. **Activate conda environment:**
   ```bash
   conda activate discoverchurn_env
   ```

## Esecuzione

1. **Execute Streamlit app:**
   ```bash
   streamlit run app.py
   ```

2. **Open the Browser to the web address provided by the app:**
   Streamlit will print a URL in the terminal; open it in your web browser.

## Dati

The dataset for this project is `Churn_Modelling.csv`, based on a bank's customer data from Spain, France, and Germany.

## Modello

The project compares five different machine learning models:
- **Random Forest**
- **Gradient Boosting**
- **Logistic Regression**
- **Support Vector Machine (SVM)**
- **K-Nearest Neighbors (KNN)**

Hyperparameter tuning has been applied to **Random Forest** and **Gradient Boosting** to improve performance. Additionally, **SMOTE** has been used to balance the dataset for **Gradient Boosting**, and its performance has been compared with **XGBoost**.

- The code to train these models is stored in `temp.py`
- The serialized **Gradient Boosting with SMOTE** model is stored in `gbm_smote_model.pkl`
- The serialized scaler is stored in `scaler.pkl`

## App

The Streamlit app allows users to insert client data and get a **churn probability prediction**. The app also includes **interactive visualizations**.

## Riassunto

The summary of my data science project can be seen in `churn_prediction.xlsx`. This file includes:
- A comparison among the five models: **Random Forest, Gradient Boosting, Logistic Regression, SVM, and KNN**
- The implementation of **hyperparameter tuning** and **SMOTE** methods applied to **Random Forest** and **Gradient Boosting**
- A comparison between **Gradient Boosting with SMOTE** and **XGBoost**
- An overview of the correlation between features and targets with different graphs
- **A geographical map visualization** showing the relationship between customer locations and churn probability

## Autore

Martina Bavetta

## Licenza

This project is released under the **MIT License**.





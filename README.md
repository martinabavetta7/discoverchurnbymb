# discoverchurnbymb
This repository showcases a data science project focusing on churn prediction. It features a Machine Learning model and an interactive Streamlit app for real-time predictions and visualizations. Specifically, it is based on a dataset containing information about male and female bank clients from Spain, Germany and France. The focus is the prediction of churn in dependence of specific features, namely age, gender, Credit Score, tenure, balance, number of products, the possession of a credit card, activity, salary and geography.
## Installazione
1.  **Clone the repository:**
 ```bash




    git clone [https://github.com/martinabavetta7/discoverchurnbymb.git](https://github.com/martinabavetta7/discoverchurnbymb.git)
    ```
2.   **Create a conda environment:**
```bash




   conda env create -f environment.yml
```
3.  **Activate conda environment:**
    ```bash



    
    conda activate discoverchurn_env
  ```  
## Esecuzione
1.  **Execute Streamlit app:**
   ```bash




    streamlit run app.py
    ```
2.  **2.  **Open the Browser to the web address provided by the app:**




*Streamlit will print an URL in the terminal, so, open it into your web browser.***
## Dati




The dataset for this project is "Churn\_Modelling.csv", based on a based on a bank's customer data from Spain, France and Germany.

## Modello




The model for this project is Gradient Boosting Classifier trained with SMOTE to balance data. The code to train this model is stored into `temp.py`; the serialised model is stored into the corresponding file  `gbm_smote_model.pkl`, while, the serialised scaler is stored into the file named`scaler.pkl`.
## App




Streamlit app allows users to insert client data and get a churn probability prevision. The app also includes interactive visualizations.
## Riassunto




The summary of my data science project can be seen by opening the file 'churn_prediction.xlsx': this file shows the comparison among different models, the implementation of hyperparameters tuning and SMOTE methods applied to Random Forest and Gradient Boosting model (the most performant models), then, a comparison between Gradient Boosting model with SMOTE and XGBoost, at the end, an overview of the correlation between features and targets with different graphs and a map. 
## Autore




Martina Bavetta
## Licenza




This project is released under the MIT License.

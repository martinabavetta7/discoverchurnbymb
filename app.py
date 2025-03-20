import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
import plotly.express as px

# Carica il modello e lo scaler
with open('gbm_smote_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

label_encoder = LabelEncoder()

# Interfaccia utente
st.title('Predizione Churn')

# Barra laterale per i controlli
st.sidebar.header('Controlli Input Cliente')

credit_score = st.sidebar.number_input('Credit Score', min_value=300, max_value=850, value=650)
gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
age = st.sidebar.number_input('Age', min_value=18, max_value=100, value=30)
tenure = st.sidebar.number_input('Tenure', min_value=0, max_value=10, value=5)
balance = st.sidebar.number_input('Balance', min_value=0.0, value=0.0)
num_of_products = st.sidebar.number_input('Number of Products', min_value=1, max_value=4, value=2)
has_cr_card = st.sidebar.selectbox('Has Credit Card', [1, 0])
is_active_member = st.sidebar.selectbox('Is Active Member', [1, 0])
estimated_salary = st.sidebar.number_input('Estimated Salary', min_value=0.0, value=50000.0)
geography = st.sidebar.selectbox('Geography', ['France', 'Germany', 'Spain'])

# Preprocessing
gender_encoded = label_encoder.fit_transform([gender])[0]
geography_germany = 1 if geography == 'Germany' else 0
geography_spain = 1 if geography == 'Spain' else 0

input_data = pd.DataFrame([[credit_score, gender_encoded, age, tenure, balance, num_of_products, has_cr_card, is_active_member, estimated_salary, geography_germany, geography_spain]],
                          columns=['CreditScore', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Geography_Germany', 'Geography_Spain'])

input_scaled = scaler.transform(input_data)

# Predizione
if st.button('Predici Churn'):
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[:, 1]

    # Visualizzazione dei risultati con metriche
    if prediction[0] == 1:
        st.metric(label="Rischio Churn", value="Alto", delta=f"{probability[0]:.2f}")
        st.warning(f'Il cliente è a rischio di churn (probabilità: {probability[0]:.2f})')
    else:
        st.metric(label="Rischio Churn", value="Basso", delta=f"{probability[0]:.2f}")
        st.success(f'Il cliente non è a rischio di churn (probabilità: {probability[0]:.2f})')

    # Grafico interattivo con Plotly (corretto)
    df_input = input_data.copy()
    df_input['Churn Probability'] = probability[0]
    df_plot = pd.DataFrame({'Feature': df_input.columns, 'Value': df_input.iloc[0].values})
    fig = px.bar(df_plot, x='Feature', y='Value', title='Caratteristiche del Cliente e Probabilità di Churn')
    st.plotly_chart(fig)
    
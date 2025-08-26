import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Load the trained model

@st.cache_resource
def load_models():
    model = load_model('model.h5', safe_mode=False)
    OHE_encoder_geo, label_encoder_gender, scaler = None, None, None
    # Load the encoders and scaler
    with open('one_hot_encoder_geo.pkl','rb') as file:
        OHE_encoder_geo = pickle.load(file)
    with open('label_encoder_gender.pkl','rb') as file:
        label_encoder_gender = pickle.load(file)
    with open('standard_scaler_x.pkl','rb') as file:
        scaler = pickle.load(file)
    return model, OHE_encoder_geo, label_encoder_gender, scaler
      
model, OHE_encoder_geo, label_encoder_gender, scaler = load_models()      
## streamlit app
st.title(model.summary())
st.title('Customer Churn Prediction')

# User input
geography = st.selectbox('Geography', OHE_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode 'Geography'
# geo_encoded = OHE_encoder_geo.transform([[geography]])
geo_encoded_df = pd.DataFrame(OHE_encoder_geo.transform([[geography]]),columns=OHE_encoder_geo.get_feature_names_out())

x1 = pd.concat([input_data, geo_encoded_df],axis=1)
st.write(x1)
x1 = scaler.transform(x1)
st.write(x1)
prediction_proba = model.predict(x1)[0][0]

st.write(f'Churn Probability: {prediction_proba:.2f}')

if prediction_proba > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')

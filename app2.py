from pycaret.regression import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

#loading the model
model = load_model('Pycaret_web_app')

def predict(model, input_df):
    predictions_df = predict_model(estimator=model, data=input_df)
    predictions = predictions_df['Label'][0]
    return predictions

image = Image.open('logo2.jpeg')
st.image(image, width=150)
st.title("Insurance Charges Prediction App")
image_hospital = Image.open('hospital.png')

add_selectbox = st.sidebar.selectbox(
    "How would you like to predict?",
    ("Online", "Batch"))

st.sidebar.info('This app is created to predict patient hospital charges')
st.sidebar.success('Created by Wale Adelanwa')

st.sidebar.image(image_hospital)

if add_selectbox == 'Online':
    st.text("Please Enter Your Details Here")
    age = st.number_input('Age', min_value=1, max_value=100, value=25)
    sex = st.selectbox('Sex', ['male', 'female'])
    bmi = st.number_input('BMI', min_value=10, max_value=50, value=10)
    children = st.selectbox('Children', [0,1,2,3,4,5,6,7,8,9,10])
    if st.checkbox('Smoker'):
        smoker = 'yes'
    else:
        smoker = 'no'
    region = st.selectbox('Region', ['southwest', 'northwest', 'northeast', 'southeast'])

    output=""

    input_dict = {'age' : age, 'sex' : sex, 'bmi' : bmi, 'children' : children, 'smoker' : smoker, 'region' : region}
    input_df = pd.DataFrame([input_dict])

    if st.button("Predict"):
        output = predict(model=model, input_df=input_df)
        output = '$' + str(output)
        st.success('The output is {}'.format(output))

if add_selectbox == 'Batch':

    file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])

    if file_upload is not None:
        data = pd.read_csv(file_upload)
        predictions = predict_model(estimator=model, data=data)
        st.write(predictions)
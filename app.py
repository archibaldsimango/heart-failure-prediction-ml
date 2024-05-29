import streamlit as st
import pandas as pd
import numpy as np
import pickle
import yaml
from PIL import Image
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth

def user_input_features():
    Age = st.sidebar.number_input('Enter your age: ', min_value=0, max_value=140)
    Sex  = st.sidebar.selectbox('Sex (Male: 1, Female: 0)',(0,1), )
    ChestPainType = st.sidebar.selectbox('ChestPainType (ATA:1, NAP:2, ASY:0, TA:3)',(0,1,2,3))
    Cholesterol = st.sidebar.number_input('Serum cholestoral in mg/dl: ')
    FastingBS = st.sidebar.selectbox('Fasting blood sugar (Yes:1, No:0)',(0,1))
    MaxHR = st.sidebar.number_input('Maximum heart rate achieved: ')
    ExerciseAngina = st.sidebar.selectbox('Exercise induced angina (No:1, Yes:0)',(0,1))
    Oldpeak = st.sidebar.number_input('Oldpeak ')
    ST_slope = st.sidebar.selectbox('Slope of the peak exercise ST segmen (Up:2, Flat:1, Down:0)',(0,1,2))

    data = {'Age': Age,
            'Sex': Sex, 
            'Chest Pain Type':  ChestPainType,
            'Cholesterol': Cholesterol,
            'FastingBS': FastingBS,
            'MaxHR':MaxHR,
            'ExerciseAngina':ExerciseAngina,
            'Oldpeak':Oldpeak,
            'ST_slope':ST_slope,
            }
    features = pd.DataFrame(data, index=[0])
    return features

def load_and_predict():
    input_df = user_input_features()
    st.write(input_df)
    if st.button("Predict"):
        # Reads in saved classification model
        load_clf = pickle.load(open('Logistic_model.pkl', 'rb'))

        # Apply model to make predictions
        prediction = load_clf.predict(input_df)
        prediction_proba = load_clf.predict_proba(input_df)

        st.subheader('Prediction')
        st.write(prediction)

        st.subheader('Prediction Probability')
        st.write(prediction_proba)
    
# Loading config file
with open('./config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

# Creating the authenticator object
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'], 
    config['cookie']['key'], 
    config['cookie']['expiry_days'],
    config['preauthorized']
)

# creating a login widget
name, authentication_status, username = authenticator.login('Login', 'main')

if authentication_status:
    authenticator.logout('Logout', 'main')

    # page_bg_img = '''
    # <style>
    # body {
    # background-image: url("images/background.jpg");
    # background-size: cover;
    # }
    # </style>
    # '''
    # st.markdown(page_bg_img, unsafe_allow_html=True)
    # # bg_image = Image.open('images/background.jpg')
    # # st.image(bg_image, use_column_width=True)

    st.write(f'Welcome *{name}*')

    st.write("""
    # Heart Disease Prediction System

    This system predicts If a patient has a heart disease

    """)
    st.sidebar.header('User Input Features')
    load_and_predict()

elif authentication_status is False:
    st.error('Username/password is incorrect')
elif authentication_status is None:
    st.warning('Please enter your username and password')
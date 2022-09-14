import numpy as np 
import pandas as pd
import pickle
import streamlit as st
loaded_model=pickle.load(open('trained_model.sav','rb'))
def add_bg_from_url():
    st.markdown(f"""
            <style>
            .stApp{{
                background-image: url("https://images.template.net/wp-content/uploads/2015/08/Elegant-Paper-Background-Free-Download.png?width=480");
                background-attachment: fixed;
                background-size: cover
                }}
            </style>
            """,unsafe_allow_html=True)

def fake_news_prediction(input_data):
    l0=loaded_model[0]
    l1=loaded_model[1]
    new=[input_data]
    tnew=l1.transform(new)
    prediction=l0.predict(tnew)
    if (prediction=='TRUE'):
        return "The news is not fake"
    else:
        return "The news  is fake"
def main():
    st.title("FAKE NEWS PREDICTION USING ML")
    st.subheader("Please State The Headline of The News")
    input_data=st.text_input("Type here 👇")
    add_bg_from_url()
    Label=""
    if st.button("Prediction Result"):
        Label=fake_news_prediction(input_data)
    st.subheader(Label)
    st.caption("Predicting Fake News Using Natural Language Processing (Machine Learning) Based on User Input")

if __name__=="__main__":
    main()
        
        
        
        

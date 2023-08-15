import pandas as pd
import numpy as np
import pickle
import streamlit as st

loaded_model=pickle.load(open("c:/users/bhima/Documents/Excelr/project/project-1/stream/trained_model.sav", 'rb'))

def Random_Forest(input_data):
    input_data_asarray=np.asarray(input_data)
    input_data_reshaped=input_data_asarray.reshape(1,-1)
    prediction=loaded_model.predict(input(input_data_reshaped)
    return prediction
                                    
def main():
    st.title("Bike Count prediction")
    instant=st.text_input("instant")
    season=st.text_input("season")
    mnth=st.text_input("mnth")
    holiday=st.text_input("holiday")
    weekday=st.text_input("weekday")
    workingday=st.text_input("working")
    weathersit=st.text_input("weathersit")
    temp=st.text_input("temp")
    hum=st.text_input("hum")
    windspeed=st.text_input("windspeed")
    casual=st.text_input("casual")
    registered=st.text_input("registered")
    
    empty = " "
     
    if st.button("Count Prediction"): 
        empty=Random_Forest([instant,season,mnth,holiday,weekday,workingday,temp,hum,windspeed,casual,registered]) 
 st.success(empty)
                                    
if__name=='__main__':
   main()
                                 
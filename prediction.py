

import streamlit as st
import datetime

import pandas as pd
import warnings
from pickle import load

warnings.filterwarnings('ignore')


df = pd.read_csv('Gold_data.csv')

# load the model from disk
loaded_model = load(open('gold_prediction_model.sav', 'rb'))


def main():
    st.title('Model Deployment: Timeseries Forecasting')
    
    s = datetime.date(2021,12,23)
    e = st.date_input("Enter the ending Date to Predict the Gold Prices")
    diff=( (e-s).days+1)
    
    
    if st.button("PREDICT"):
        index_future_dates=pd.date_range(start= s ,end= e)
        pred=loaded_model.forecast(diff).rename('Price')
        pred.index=index_future_dates.rename('date')
        df = pd.DataFrame(pred)
        
        st.dataframe(df)
        st.line_chart(df)
        
        
if __name__ == '__main__':
    main() 
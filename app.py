# Imports
import streamlit as st
# Write a page title
from datetime import date

from streamlit import cli as stcli 
import sys
from streamlit import cli as stcli

import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go

from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
from transformers import pipeline


st.markdown("""
<style>
form {
  unicode-bidi:bidi-override;
  direction: RTL;
}
</style>
    """, unsafe_allow_html=True)


def translate(text):
    return GoogleTranslator(source='english', target='arabic').translate(text)
def anlyze(text):
    finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone',num_labels=3)
    tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
    nlp = pipeline("sentiment-analysis", model=finbert, tokenizer=tokenizer)


    return nlp(text)





def main():
   
	st.title("اثمار لتحليل الاسهم")
	
   

	menu = ["رصد تأثير الخبر","تحليلات و توقعات"]
	choice = st.sidebar.selectbox("القائمة",menu)
    

	if choice == "رصد تأثير الخبر":
		st.subheader("رصد تأثير الخبر")
		with st.form(key='nlpForm'):
			raw_text = st.text_area("ادخل الخبر")
			submit_button = st.form_submit_button(label='حلل')

		# layout
		col1,col2 = st.columns(2)
		if submit_button:

			with col1:
				st.info("النتائج")
				sentiment = anlyze(raw_text)
				st.write(sentiment)      
	else:
         st.subheader("تحليلات و توقعات")
         START = "2015-01-01"
         TODAY = date.today().strftime("%Y-%m-%d")
         

         stocks = ('7010.SR', '2222.SR')
         selected_stock = st.selectbox('اختر السهم', stocks)

         n_years = st.slider('سنوات التوقع:', 1, 7)
         period = n_years * 365


         @st.cache
         def load_data(ticker):

            data = yf.download(ticker, START, TODAY)
            data.reset_index(inplace=True)
            return data

	
         data_load_state = st.text('تحميل البيانات.....')
         data = load_data(selected_stock)
         data_load_state.text('تم تحميل البيانات!')

         st.subheader('البيانات خام')
         st.write(data.tail())

         # Plot raw data
         def plot_raw_data():
             fig = go.Figure()
             fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
             fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
             fig.layout.update(title_text='البيانات الزمنية', xaxis_rangeslider_visible=True)
             st.plotly_chart(fig)
	
         plot_raw_data()

# Predict forecast with Prophet.
         df_train = data[['Date','Close']]
         df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

         m = Prophet()
         m.fit(df_train)
         future = m.make_future_dataframe(periods=period)
         forecast = m.predict(future)

# Show and plot forecast
         st.subheader('التوقع')
         st.write(forecast.tail())
    
         st.write(f'التوقع ل {n_years} سنوات')
         fig1 = plot_plotly(m, forecast)
         st.plotly_chart(fig1)

         st.write("عناصر التوقع")
         fig2 = m.plot_components(forecast)
         st.write(fig2)
    
        
   




if __name__ == '__main__':

 
    main()
  

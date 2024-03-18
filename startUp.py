import streamlit as st
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

#st.title('START UP PROJECT')
#st.subheader('Built By Gomycode Daintree')

st.markdown("<h1 style = 'color: #5E1675; text-align: center; font-family: helvetica'>STARTUP PROFIT PREDICTOR</h1>", unsafe_allow_html = True)
st.markdown("<h4 style = 'margin: -30px; color: #337357; text-align: center; font-family: cursive '>Built By PluralCode Data Science Cohort</h4>", unsafe_allow_html = True)
st.markdown("<br>", unsafe_allow_html= True)

st.image('pngwing.com (1).png')

st.header('Project Background Information', divider = True)
st.write("The overarching objective of this ambitious project is to meticulously engineer a highly sophisticated predictive model meticulously designed to meticulously assess the intricacies of startup profitability. By harnessing the unparalleled power and precision of cutting-edge machine learning methodologies, our ultimate aim is to furnish stakeholders with an unparalleled depth of insights meticulously delving into the myriad factors intricately interwoven with a startup's financial success. Through the comprehensive analysis of extensive and multifaceted datasets, our mission is to equip decision-makers with a comprehensive understanding of the multifarious dynamics shaping the trajectory of burgeoning enterprises. Our unwavering commitment lies in empowering stakeholders with the indispensable tools and knowledge requisite for making meticulously informed decisions amidst the ever-evolving landscape of entrepreneurial endeavors.")

st.markdown("<br>", unsafe_allow_html= True)
st.markdown("<br>", unsafe_allow_html= True)

data = pd.read_csv('startUp.csv')
st.dataframe(data)

#Input User Image
st.sidebar.image('pngwing.com (2).png', caption = 'Welcome User')

#Apply space in the sidebar
st.markdown("<br>", unsafe_allow_html= True)
st.markdown("<br>", unsafe_allow_html= True)

#Declare user Input variables
st.sidebar.subheader('Input Variables', divider = True)
rd_spend = st.sidebar.number_input('Research And Development Expense')
admin = st.sidebar.number_input('Administrative Expense')
mkt = st.sidebar.number_input('Marketing Expense')

#display the users input
input_var = pd.DataFrame()
input_var['R&D Spend'] = [rd_spend]
input_var['Administration'] = [admin]
input_var['Marketing Spend'] = [mkt]

st.markdown("<br>", unsafe_allow_html= True)
#display the users input variable
st.subheader('Users Input Variable', divider = True)
st.dataframe(input_var)

#Import the scalars
admin_scaler = joblib.load('Administration_scaler.pkl')
mkt_scaler = joblib.load('Marketing Spend_scaler.pkl') 
rd_spend_scaler = joblib.load('R&D Spend_scaler.pkl')

  #transform the users input with the imported scalers
input_var['R&D Spend'] = rd_spend_scaler.transform(input_var[['R&D Spend']])
input_var['Administration'] = admin_scaler.transform(input_var[['Administration']])
input_var['Marketing Spend'] = mkt_scaler.transform(input_var[['Marketing Spend']])

#st.dataframe(input_var) 

#import the model
model = joblib.load('startUpModel.pkl')
predicted = model.predict(input_var)

st.markdown("<br>", unsafe_allow_html= True)

# if st.button('Predict Profit'):
#     st.balloons()
#     st.success(f'The predicted profit for your organisation is: {predicted[0].round(2)}')

#Creating prediction and interpretation tab
prediction, interprete = st.tabs(["Model Prediction", "Model Interpretation"])
with prediction:
    if prediction.button('Predict Profit'):
        prediction.snow()
        prediction.success(f'The predicted profit for your organisation is {predicted[0].round(2)}')


#inter.write("this is tab 2")
with interprete: 
    intercept = model.intercept_
    coef = model.coef_
    interprete.write(f'A percentage increase in Research and Development Expense makes Profit to incease by {coef[0].round(2)} naira')
    interprete.write(f'A percentage increase in Administratin Expense makes Profit to incease by {coef[0].round(2)} naira')
    interprete.write(f'A percentage increase in Marketing Expense makes Profit to incease by {coef[0].round(2)} naira')
    interprete.write(f'the value of Profit when none of these expenses were made is {intercept.round(2)} naira')


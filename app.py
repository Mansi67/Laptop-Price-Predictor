import streamlit as st
import pickle
import numpy as np


#import the model

pipe= pickle.load(open('C:/Users/mansi/Desktop/CSv/laptop_price_predictor/pipe.pkl','rb'))
df= pickle.load(open('C:/Users/mansi/Desktop/CSv/laptop_price_predictor/df.pkl','rb'))

st.title("Laptop Price Predictor")

# brand
company= st.selectbox('Brand',df['Company'].unique())

# type of laptop
type=st.selectbox('Type',df['TypeName'].unique())

# Ram
ram=st.selectbox('Ram (GB)', [2,4,6,8,12,16,24,32,64])

#Weight
weight=st.number_input('Weight of Laptop')

#TouchScreen
touchscreen=st.selectbox('TouchScreen', ['Yes', 'No'])

#Full HD
hd=st.selectbox('Full HD', ['Yes', 'No'])

#IPS Panel
ips=st.selectbox('IPS Panel',['Yes', 'No'])

#Screensize
screen_size=st.number_input('Screen size in Inches')

#Resolution
resolution=st.selectbox('Screen Resolution',['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'] )

#GHz
ghz=st.number_input('Speed in GHz')

#CPU
cpu=st.selectbox('CPU Brand',df['Cpu brand'].unique())

#HDD Drive
hdd=st.selectbox('HDD(in GB)',[0,128,256,512,1024,2048])

#ssd
ssd= st.selectbox('SSD(in GB)',[0,8,128,256,512,1024])

# gpu
gpu=st.selectbox('GPU',df['Gpu brand'].unique())

#operating system
os = st.selectbox('OS',df['os'].unique())

if st.button('Predict Price'):
    # query
    ppi = None
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0

    if hd == 'Yes':
        hd = 1
    else:
        hd = 0

    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2) + (Y_res**2))**0.5/screen_size
    query = np.array([company,type,ram,weight,touchscreen,hd,ips,ppi,ghz,cpu,hdd,ssd,gpu,os], dtype=object)

    query = query.reshape(1,14)
    st.title("The predicted price of this configuration is " + str(int(np.exp(pipe.predict(query)[0]))))


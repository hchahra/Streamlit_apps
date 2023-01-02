msg='Hell World'
print(msg )
msg.lower()
msg.capitalize()
msg.upper()
import streamlit as st
st.header('Hello World')

import pandas as pd
df=pd.DataFrame({'x':[1,2,3]})

st.write(df)

col1, col2= st.columns([1,2])
with col1:
    st.header('cc')
with col2:
    st.header('hh')
st.latex('DataScience')
st.latex('Project')
from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler()
df_scaled=scaler.fit_transform(df)
st.write(df_scaled)
st.write(df)
col3, col4= st.columns(2)
with col3:
    st.latex(r''' x1+x2+x3=y''')
with col4:
    st.latex(r''' Vente=f(prix)+ {\epsilon}''')
    st.latex(r' R^2 = 98 {\%} ')
    st.latex(r'''Fisher=''')
    st.latex(r''' Taux\ de\ précision\ = 97 {\%}''')
    st.latex(r''' Welcome\ to\ my\ datascience\ project''')

import plotly.express as px
fig1=px.line(df)

    
with col4:
    st.plotly_chart(fig1)




x=st.sidebar.slider('sdffq',1,10,2)
st.sidebar.latex(x**2*x)


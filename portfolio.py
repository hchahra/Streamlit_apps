# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 20:15:42 2022

@author: hamit
"""
#******************************************************************** Import libraries


import streamlit as st
from PIL import Image
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import make_column_transformer
import numpy as np
from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.graph_objs as go
import plotly.express as px
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.metrics import classification_report_imbalanced
import pyautogui
from streamlit_lottie import st_lottie
import json
import requests
from streamlit_option_menu import option_menu



def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()








#********************************************************************************************Import database


df=pd.read_csv(r"C:\Users\hamit\Desktop\Project_final\auto.csv")


#********************************************************************************************Import picture
                                                                            







opt= st.sidebar.radio('Bienvenus dans mon Portfolio:',['Qui suis-je ?',  'Project', 'Contact'])
if opt=='Qui suis-je ?':
    st.write("**Une Data Analyst en herbe , pationnée par le domaine de la programmation, les Statistiques, ainsi que l'Intelligence Artificielle.**")
    edu=pd.DataFrame({"Master II Economie Quantitative": [ "Reconnaisance ENIC NARIC Niveau 7", "Université de Bejaia", "09/2017-07/2019"], 'RNCP Data Analyst': [" Niveau 6","Ironhack Paris","07/2022-09/2022"]}).reset_index(drop=True)
    st.write(edu)
    st.title('CV')
    uploaded_file = st.file_uploader('', type="pdf")
    if uploaded_file is not None:
        file = extract_data(uploaded_file)
    
#elif opt== 'Formations':
    #edu=pd.DataFrame({"Master II Economie Quantitative": [ "Reconnaisance ENIC NARIC Niveau 7", "Université de Bejaia", "09/2017-07/2019"], 'RNCP Data Analyst': [" Niveau 6","Ironhack Paris","07/2022-09/2022"]}).reset_index(drop=True)
    #st.write(edu)
   
    
   

elif opt== 'Project':
    
    opt1= st.sidebar.radio('Utilisation du  Machine Learning',['Segmentation des assurés auto et classification de la charge de risque', 'Ce présent portfolio'] )
    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
    if opt1== 'Segmentation des assurés auto et classification de la charge de risque':
        st.header('Segmentation des assurés auto et Prévision de la charge de risque')
        opt2= st.radio('Plan',['Présentation', 'Base de données', 'Segmentation des assurés', 'Prévision de la charge de risque', "Conclusion"])
        
        
        
        
        
        
#****************************************************************************************Presentation  





      
        if opt2== 'Présentation':
            st.header('Présentation')
            st.write("**Ce travail fut mon projet final chez Ironhack où j'ai utilisé différentes techniques d'apprentissage automatique afin de regrouper les assurés et de prédire leur classe de risque**")
            col1, col2, col3=st.columns([2,2,1]) 
            with col2:
                lottie_hello = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_ic37y4kv.json")

                st_lottie(
             lottie_hello,
             speed=1,
              reverse=False,
               loop=True,
               quality="low", # medium ; high
                
                height=None,
                width=None,
               key=None,)
            with col1:
                
                st.write("**Les compétences déployées pour l'aboutissement de ce project sont:** ")
                st.write("* Collecte des données")
                st.write('* Nettoyage des données')
                st.write("* Imputation des valeurs aberrantes")
                st.write("* Data Visualization en utilisant **Tableau**")
                st.write('* Apprentissage non supervisé en utilisant **k-prototypes technique**')
                st.write("* Analyse en composantes principales")
                st.write("* Apprentissage supervisé en utilisant **Imbalanced Random forest Classifier**")
            with col3:
                 st.write('Logiciel Utilisé: **Python**')
        if opt2== 'Base de données':
            st.subheader('The Dataset as presented has been cleaned. If you are interested by the cleaning steps I will be happy to share it with you')
            st.header('Dataframe')
            st.write(df)
            
            
            
            
            
            
            
            
#************************************************************************************Unsupervised Learning






            
        elif opt2== 'Segmentation des assurés':
            opt3 = st.selectbox(
    'What are your favorite colors',
    ['Choix des variables', 'Base de données', 'La Recherche du nombre de segments optimal'])

            
            
            
            
            
            
            
            
            
            if opt3== 'Choix des variables':
                st.subheader('I) Choix des variables')
            
                st.write("Les variables qu'on a pris compte pour la classification des assurés de façon non supervisée sont:")
                col1, col2=st.columns(2)
                with col1:
                    st.write("**-Le montant de la charge de risque**")
                    st.write("**-L'option de la limitattion du kilometrage**")
                    st.write("**-L'usage du véhicule**")
                    st.write("**-Type d'énergie**")
                    st.write("**-L'âge du véhicule**")
                with col2:
                        st.write("**-L'âge du conducteur**")
                        st.write("**-Le nombre d'années depuis la  délivrance du permis**")
                        st.write("**-Le garage**")
                        st.write("**-La vitesse maximale du véhicule**")
                        st.write("**-La classe de prix du véhicule**")
            elif opt3== 'Base de données':
                st.subheader(' II) Base de données')
            
                auto=df[['Amount_paid',
       'Limited_Milesage_option', 'Vehicle_use',
       'Energy_type', 'Vehicle_age', 'Driver_age', 'License_issuance',
        'Garage', 'Max_speed',
       'Price_class_vehicle']]
            
            
                transformer=make_column_transformer((MinMaxScaler(),['Amount_paid','Vehicle_age', 'Driver_age',  'License_issuance', 'Max_speed']))
                a=transformer.fit_transform(auto)
                a=pd.DataFrame(a)
                a.columns=['Amount_paid','Vehicle_age', 'Driver_age',  'License_issuance', 'Max_speed']
                
                b=auto[['Limited_Milesage_option', 'Vehicle_use', 'Energy_type', 'Garage',  'Price_class_vehicle' ]]
            
                auto_scal=pd.concat([a,b], axis=1)
                auto_scal.head()
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader('Standardiation des columns numériques')
                    st.write(auto_scal)
                    # Get the position of the categorical columns
                    catColumnsPos = [auto_scal.columns.get_loc(col) for col in list(auto_scal.select_dtypes('object').columns)]
                    auto_array=auto_scal.to_numpy()
                with col2:     
                        st.subheader("Conversion du dataset en forme matricielle")
                        st.write(auto_array)
            elif opt3== 'La Recherche du nombre de segments optimal':
            
            
#*********************************************************************** The optimal clusters
           
                st.subheader('III) La Recherche du nombre de segments optimal*')
                st.write("*A fin de déterminer le nombre de seglentes (clusters) optimal, nous allons utliser la méthode du Coude (Elbow method)")


           
        
            
            
            
            
            
            
#***************************************************************************************Supervised Learning



#************************************************************PCA


            
        elif opt2== 'Prévision de la charge de risque':
            auto=pd.read_csv(r'C:\Users\hamit\Desktop\Project_final\cluster.csv')
            x=auto[[ 'Limited_Milesage_option', 'Vehicle_use', 'Energy_type',
       'Vehicle_age', 'Driver_age', 'License_issuance', 'Garage', 'Max_speed',
       'Price_class_vehicle']]
            y=auto[['clusters']]
            numerical_features=['Vehicle_age', 'Driver_age',  'License_issuance', 'Max_speed']
            categorical_features=['Limited_Milesage_option', 'Vehicle_use', 'Energy_type', 'Garage',  'Price_class_vehicle' ]
            x[categorical_features]
            transformer2=make_column_transformer( (LabelEncoder(),categorical_features))
            transformer1=make_column_transformer((StandardScaler(), ['Vehicle_age', 'Driver_age',  'License_issuance', 'Max_speed']))
            a=transformer1.fit_transform(x)
            a=pd.DataFrame(a)
            a.columns=['Vehicle_age', 'Driver_age',  'License_issuance', 'Max_speed']
            a
            b=x[categorical_features]
            model=LabelEncoder()
            b=x[categorical_features].apply(LabelEncoder().fit_transform)
            b
            x=pd.concat([a,b], axis=1)
            x
            y
            pca = PCA(n_components=3)
            pca.fit_transform(x)
            explained_variance = pca.explained_variance_ratio_
            explained_var= pd.DataFrame(explained_variance, index=['PC 1', 'PC 2', 'PC 3'])
            explained_var.columns=['Total Explained Variance']
            explained_var
            
            components = pca.fit_transform(x)
            total_var = pca.explained_variance_ratio_.sum() * 100
            fig = px.scatter_3d(
    components, x=0, y=1, z=2,
    title=f'Total Explained Variance: {total_var:.2f}%', color=auto['clusters'],
    labels={'0': 'PC 1', '1': 'PC 2', '2': 'PC 3'}
)
            st.plotly_chart(fig)
            
            x_train, x_test, y_train, y_test= train_test_split(x,y, test_size=0.2)
            st.write(f'{x_train.shape},{x_test.shape}' )
            
            
            
            
#************************************************************Imbalanced Random Forest Classifier

            
            clf = BalancedRandomForestClassifier(max_depth=3, random_state=0)
            clf.fit(x_train, y_train)  
            BalancedRandomForestClassifier(...)
            y_true=y_test
            y_pred=clf.predict(x_test)
            st.write(classification_report_imbalanced(y_true, y_pred))
            tab=pd.DataFrame({ ' Prediction':[0.89,0.91,0.96], 'Recall':[0.92,0.93,0.92], 'f1':[0.90,0.92,0.94]}, index=['Cluster 0', 'Cluster 1', 'Cluster 2'])
            st.write(tab)
        #elif opt2== 'Conclusion':
            




           
            
            






           
            
            
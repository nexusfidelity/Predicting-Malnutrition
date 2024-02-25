import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import folium
from streamlit_folium import st_folium
from pycaret.classification import *


st.set_option('deprecation.showPyplotGlobalUse', False)

df = pd.read_csv('dataset/total_0-5_children.csv')
single_map_var = folium.Map(location=[12.77631195457266, 122.8], zoom_start=6, tiles='OpenStreetMap',zoom_control=False,attributionControl=False)
geojson = gpd.read_file('dataset/final_heatmap.json')
stunting_model = load_model('models/stunting_model')
wasting_model = load_model('models/wasting_model')
weight_model = load_model('models/weight_status_model')

def single_map(dataframe,geo):
    df_heatmap = geo.merge(dataframe,left_on='name',right_on='Province', how="inner")
    columns = ['name', 'count']
    heatmap_key = 'feature.properties.name'
    
    folium.Choropleth(
            geo_data='dataset/final_heatmap.json',
            data=df_heatmap,
            columns=columns,
            key_on= heatmap_key, #Here we grab the geometries/county boundaries from the geojson file using the key 'coty_code' which is the same as county fips
            #threshold_scale=[], #use the custom scale we created for legend
            fill_color='YlOrRd',
            nan_fill_color="White", #Use white color if there is no data available for the county
            fill_opacity=0.7,
            line_opacity=0.2,
            legend_name='Counts of variable ', #title of the legend
            highlight=True,
            line_color='black').add_to(single_map_var) 
    
    # #Add Customized Tooltips to the map
    folium.features.GeoJson(
        data=df_heatmap,
        smooth_factor=2,
        style_function=lambda x: {'color':'black','fillColor':'transparent','weight':0.5},
        tooltip=folium.features.GeoJsonTooltip(
            fields=columns,
            aliases=[
                'Province: ',
                'counts : '
                ],
            localize=True,
            sticky=False,
            labels=True,
            style="""
                background-color: #F0EFEF;
                border: 2px solid black;
                border-radius: 3px;
                box-shadow: 3px;
            """,
            max_width=800,),
        highlight_function=lambda x: {'weight':3,'fillColor':'grey'},).add_to(single_map_var)
    
    folium.plugins.Fullscreen(
    position="bottomright",
    title="Expand me",
    title_cancel="Exit me",
    force_separate_button=True,).add_to(single_map_var)

    
    st_folium(single_map_var,width='100%')

st.set_page_config(page_title='Predicting Malnutrition', page_icon='🤖', layout="centered", initial_sidebar_state="collapsed", menu_items=None)

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(['FNRI','WHO standards','Dataset','EDA','Choropleth map','ML training', 'Deployment'])

with tab1:
    st.header('DOST Food and Nutrition Research Institute')
    
    st.image('images/fnri_building.jpg')
    
    st.subheader('FNRI')
    
    st.write('mandates of FNRI')
    
    st.write('1. monitor the nutritional status of the Philippines')
    
    st.write('2. research and develop products that concerns food and nutrition')
    
    st.write('3. diffuse/transfer said technologies to the public')
    
    st.subheader('National Nutrition Survey')
    
    st.write('for several years, FNRI conducts surveys around the country to monitor the nutritional status in the country. These surveys accumulate to form the NNS.')
    
    st.write('The NNS were conducted from 2018-19, 2015, 2013, 2011, 2008, 2005, 2003, etc.. up to 1978')
    
    st.write('For the purpose of this study, the anthropometric dataset of years 2018-19, 2015, 2013, 2011, 2008 are used ')
    
    st.image('images/surveyors.png')
    
with tab2:
    st.header('World Health Organization Standards')
    
    st.write('These standards were developed using data collected in the WHO Multicentre Growth Reference Study')
    
    st.link_button("Go to WHO standards", "https://www.who.int/tools/child-growth-standards")
    
    st.subheader('Stunting')
    st.write('Stunting is defined as low height-for-age. It is the result of chronic or recurrent undernutrition, usually associated with poverty, poor maternal health and nutrition, frequent illness and/or inappropriate feeding and care in early life. Stunting prevents children from reaching their physical and cognitive potential.')
    col1, col2 = st.columns(2)
    with col1:
        st.image('images/weight-boy.png')
    with col2:
        st.image('images/weight-girl.png')
    
    st.subheader('Wasting')
    st.write('Wasting is defined as low weight-for-height. It often indicates recent and severe weight loss, although it can also persist for a long time. It usually occurs when a person has not had food of adequate quality and quantity and/or they have had frequent or prolonged illnesses.')
    col1, col2 = st.columns(2)
    with col1:
        st.image('images/weight-for-height-boys.png')
    with col2:
        st.image('images/weight-for-height-girls.png')
    
    st.subheader('Overweight and Underweight')
    
    col1, col2 = st.columns(2)
    with col1:
        st.image('images/stunting-boys.png')
    with col2:
        st.image('images/stunting-girls.png')

with tab3:
    
    st.header('Dataset')
    
    st.dataframe(df)
    
    st.header('Data preparation')
    
    st.subheader('Data cleaning')
    st.write('null values are represented as 8888 in dataset and replace with None. We let pycaret deal with the null values later')
    st.write('replace col values from primary keys to actual values in data dictionary')
    
    st.divider()
    st.link_button("Go to enutrition", "https://enutrition.fnri.dost.gov.ph/")
    
    
with tab4:
    st.header('Exploratory Data Analysis')
    
    ### Stunting
    st.subheader('Stunting')
    
    st.dataframe(df['stunted'].value_counts())
    
    dfu =df.groupby(['year'])[['sex','stunted']].value_counts().unstack()
    fig, ax = plt.subplots(figsize=(10,4))
    ax = dfu.plot(kind='barh', figsize=(10, 10), xlabel='Count', ylabel='Year, Gender', rot=0)
    ax.legend(title='distribution', bbox_to_anchor=(1, 1), loc='upper left')
    st.pyplot()
    
    ### Wasting
    st.subheader('Wasting')
    
    st.dataframe(df['wasted'].value_counts())
    
    dfu2 = df.groupby(['year'])[['sex','wasted']].value_counts().unstack()
    fig, ax2 = plt.subplots(figsize=(10,4))
    ax2 = dfu2.plot(kind='barh', figsize=(10, 10), xlabel='Count', ylabel='Year, Gender', rot=0)
    ax2.legend(title='distribution', bbox_to_anchor=(1, 1), loc='upper left')
    st.pyplot()
    
    ### Underweight and Overweight
    st.subheader('Underweight and Overweight')
    
    st.dataframe(df['weight_status'].value_counts())
    
    dfu3 = df.groupby(['year'])[['sex','weight_status']].value_counts().unstack()
    fig, ax2 = plt.subplots(figsize=(10,4))
    ax3 = dfu3.plot(kind='barh', figsize=(10, 10), xlabel='Count', ylabel='Year, Gender', rot=0)
    ax3.legend(title='distribution', bbox_to_anchor=(1, 1), loc='upper left')
    st.pyplot()
    
with tab5:
    st.header('Map')
    
    var = st.selectbox('select variable',['moderately stunted','severely stunted','moderately wasted','severely wasted','underweight','overweight','obese'], index = 0)
    
    if var == 'moderately stunted' or var == 'severely stunted':
        df_map = df[df['stunted'] == var]
        df_map = df_map.groupby(['Province'])['stunted'].value_counts()#.to_frame()
        
    if var == 'moderately wasted' or var == 'severely wasted':
        df_map = df[df['wasted'] == var]
        df_map = df_map.groupby(['Province'])['wasted'].value_counts()#.to_frame()
    
    if var == 'underweight' or var == 'overweight' or var == 'obese':
        df_map = df[df['weight_status'] == var]
        df_map = df_map.groupby(['Province'])['weight_status'].value_counts()#.to_frame()
    
    single_map(df_map,geojson)

with tab6:
    st.header('Training')
    
    st.code('''
            from pycaret.datasets import get_data
            from pycaret.classification import *
            import pandas as pd            
            ''',)
    
    st.subheader('Stunting')
    
    st.code('''
            stunting_setup = setup(data = data,  
                       target = 'stunted',
                       ignore_features = ['Region','Province','wasted','bmi','bmi range','year','weight_status'],
                       # imputation_type=None,
                       use_gpu=True,
                       fix_imbalance=True
                )
            ''',)
            
    st.code('''
            stunting_model = stunting_setup.compare_models()
            ''',)
            
    st.code('''
            evaluate_model(stunting_model)
            ''',)
            
    st.image('images/stunting-compare-models.png')
    st.image('images/stunting-confusion-matrix.png')
    st.image('images/stunting-prediction-error.png')
    st.image('images/stunting-feature-importance.png')
            
    st.subheader('Wasting')
    
    st.code('''
            df_wasting = df[df['wasted'].notna()]
            ''',)
    
    df_wasting = df[df['wasted'].notna()]
    st.dataframe(df_wasting.head())
    
    
    st.code('''
            wasting_setup = setup(data = df_wasting,  
                      target = 'wasted',
                      ignore_features = ['Region','Province','stunted','bmi','bmi range','year','weight_status'],
                      use_gpu=True,
                      fix_imbalance=True
                )
            ''',)
            
    st.code('''
            wasting_model = wasting_setup.compare_models()
            ''',)
            
    st.code('''
            evaluate_model(wasting_model)
            ''',)
    
    st.image('images/wasting-compare-models.png')
    st.image('images/wasting-confusion-matrix.png')
    st.image('images/wasting-prediction-error.png')
    st.image('images/wasting-feature-importance.png')
    
    st.code('''
            save_model(wasting_model, 'wasting_model')
            ''',)
            
    st.subheader('Underweight and Overweight')
    
    st.code('''
            weight_setup = setup(data = data,  
                     target = 'weight_status',
                     ignore_features = ['Region','Province','stunted','bmi','bmi range','year','wasted'],
                     use_gpu=True,
                     fix_imbalance=True
                )
                )
            ''',)
    
    st.code('''
            weight_status_model = weight_setup.compare_models()
            ''',)
            
    st.code('''
            evaluate_model(weight_status_model)
            ''',)
    
    st.image('images/weight-compare-models.png')
    st.image('images/weight-confusion-matrix.png')
    st.image('images/weight-prediction-error.png')
    st.image('images/weight-feature-importance.png')
    
    st.code('''
            save_model(weight_status_model, 'weight_status_model')
            ''',)
            
with tab7:
    st.header('Model Deployment')
    
    col1, col2, col3, col4 = st.columns(4)
    
    age = col1.number_input('Age (years old)',min_value=0.0,max_value=5.0, help='0-5 years old')
    sex = col2.selectbox('Gender',['boy','girl'])
    height = col3.number_input('Height (cm)')
    weight = col4.number_input('Weight (kg)')
    
    submit_button = st.button('Submit',use_container_width=True)
    
    if submit_button:
        pass
        #create df
        df_input_stunting = pd.DataFrame(columns = [
            'age',
            'sex',
            'weight',
            'height'])
        
        df_input_stunting.loc[0] = [age, sex,weight,height]
        
        df_input_stunting = df_input_stunting.replace(
            {'sex':{
            'boy':0,
            'girl':1,
            }})
        
        stunting_prediction = predict_model(stunting_model,data = df_input_stunting,verbose=False)
        
        st.write('stunting prediction: '+stunting_prediction.iloc[0,-2])
        st.write('prediction confidence: '+str((stunting_prediction.iloc[0,-1])*100))
        
        wasting_prediction = predict_model(wasting_model,data = df_input_stunting,verbose=False)
        
        st.write('wasting prediction: '+wasting_prediction.iloc[0,-2])
        st.write('prediction confidence: '+str((wasting_prediction.iloc[0,-1])*100))
        
        weight_prediction = predict_model(weight_model,data = df_input_stunting,verbose=False)
    
        st.write('weight status prediction: '+weight_prediction.iloc[0,-2])
        st.write('prediction confidence: '+str((weight_prediction.iloc[0,-1])*100))
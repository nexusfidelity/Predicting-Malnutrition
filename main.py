import streamlit as st

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(['FNRI','WHO standards','Dataset','EDA','ML training', 'Deployment'])

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
    
    st.write('For the purpose of this study, the anthropometric dataset of years 2018-19, 2015, 2013, 2011, 2008, 2005, 2003 are used ')
    
    st.image('images/surveyors.png')
    
with tab2:
    st.header('WHO standards')
    
    st.write('These standards were developed using data collected in the WHO Multicentre Growth Reference Study')
    
    st.link_button("Go to WHO standards", "https://www.who.int/tools/child-growth-standards")
    
    st.subheader('Stunting')
    
    col1, col2 = st.columns(2)
    with col1:
        st.image('images/weight-boy.png')
    with col2:
        st.image('images/weight-girl.png')
    
    st.subheader('Wasting')
    
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
    
    
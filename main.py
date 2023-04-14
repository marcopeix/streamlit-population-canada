import streamlit as st
import pandas as pd
import numpy as np
import urllib.request
import json
import plotly.express as px
import matplotlib.pyplot as plt

st.set_page_config('Homepage', page_icon="🏡")

with st.sidebar:
    st.markdown("# Hello 👋")
    st.markdown("## Welcome to this app on visualizing and forecasting the quarterly population in Canada. Of course, real-life data is used. Have fun 🎉!")

st.title('Quarterly Population Estimates in Canada')
st.subheader('From Q3 1991 to Q1 2023')
st.markdown("Source table can be found [here](https://www150.statcan.gc.ca/t1/tbl1/en/cv.action?pid=1710000901)")

@st.cache_data
def read_data():
    URL = "https://raw.githubusercontent.com/marcopeix/streamlit-population-canada/master/data/quarterly_canada_population.csv"
    df = pd.read_csv(URL, index_col=0, dtype={'Quarter': str, 
                                          'Canada': np.int32,
                                         'Newfoundland and Labrador': np.int32,
                                          'Prince Edward Island': np.int32,
                                          'Nova Scotia': np.int32,
                                          'New Brunswick': np.int32,
                                          'Quebec': np.int32,
                                          'Ontario': np.int32,
                                          'Manitoba': np.int32,
                                          'Saskatchewan': np.int32,
                                          'Alberta': np.int32,
                                          'British Columbia': np.int32,
                                          'Yukon': np.int32,
                                          'Northwest Territories': np.int32,
                                          'Nunavut': np.int32})

    return df, df.T

@st.cache_data
def read_geojson():
    JSON_URL = "https://raw.githubusercontent.com/marcopeix/streamlit-population-canada/master/data/canada.geojson"
    with urllib.request.urlopen(JSON_URL) as res:
        geo = json.load(res)

    return geo

df, geo_df = read_data()
if df not in st.session_state:
    st.session_state['df'] = df
geo = read_geojson()
st.write(df)

col1, col2 = st.columns(2)

quarter_option = col1.selectbox(label='Quarter', options=['Q1', 'Q2', 'Q3', 'Q4'])
year_option = col2.selectbox(label='Year', options=np.arange(1991, 2024, 1)[::-1])

try:
    
    fig = px.choropleth(
        geo_df,
        geojson=geo,
        # locations=['Canada','Newfoundland and Labrador','Prince Edward Island','Nova Scotia','New Brunswick','Quebec','Ontario','Manitoba','Saskatchewan','Alberta','British Columbia','Yukon','Northwest Territories','Nunavut'],
        locations=geo_df.index,
        featureidkey="properties.name",
        color=f"{quarter_option} {year_option}",
        color_continuous_scale='oranges'
    )

    fig.update_layout(margin={"r": 0, "t":0, "l":0, "b":0})
    fig.update_geos(fitbounds="locations", visible=True)

    st.plotly_chart(fig)
except ValueError:
    st.error('No data available for your quarter and year selection')

tab1, tab2 = st.tabs(['Line plot', 'Bar plot'])

province_option = tab1.selectbox(label='Select a province or a territory', options=df.columns[1:])
line_fig, line_ax = plt.subplots()
line_ax.plot(df[f'{province_option}'])
line_ax.set_xticks(np.arange(2, len(df), 8))
line_ax.set_xticklabels(np.arange(1992, 2024, 2))
line_ax.yaxis.set_major_locator(plt.MaxNLocator(10))
line_fig.autofmt_xdate()
tab1.subheader(f"{province_option}")
tab1.pyplot(line_fig)

@st.cache_data
def format_data_for_animated_barplot(df):
    anim_bar_df = pd.DataFrame()

    for each in df.columns[1:]:
        temp_df = pd.DataFrame({'pop': df[f'{each}'].values, 'name': f'{each}'})
        anim_bar_df = pd.concat([anim_bar_df, temp_df])

    dates = []
    qs = df.index.values
    for _ in range(13):
        dates.extend(qs)
    anim_bar_df['date'] = dates

    return anim_bar_df

anim_bar_df = format_data_for_animated_barplot(df)
bar_fig = px.bar(anim_bar_df, 
                 x='name', 
                 y='pop', 
                 color='name', 
                 animation_frame='date', 
                 animation_group='name', 
                 range_y=[0, 16000000])
tab2.plotly_chart(bar_fig)

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

st.set_page_config(
    page_title="RoadSafe Analytics Dashboard",
    layout="wide"
)

st.title("RoadSafe Analytics – Interactive Accident Dashboard")

st.markdown("""
### Project Overview

**RoadSafe Analytics** is an interactive data analytics dashboard designed to explore, analyze,  
and visualize **US road accident data** across multiple dimensions such as:

- **Geography** (State, City, Coordinates)
- **Time** (Hour, Weekday, Month, Date Range)
- **Weather Conditions**
- **Accident Severity**
- **Visibility Levels**
""")

@st.cache_data
def load_data():
    df = pd.read_parquet("roadsafe_cleaned.parquet")
    df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
    return df

df_full = load_data()

st.sidebar.header("Filters")

filter_df = df_full[
    ['State', 'TimeOfDay', 'Weather_Condition', 'Severity']
].drop_duplicates()

states = sorted(filter_df['State'].dropna().unique())
time_options = sorted(filter_df['TimeOfDay'].dropna().unique())
weather_options = sorted(filter_df['Weather_Condition'].dropna().unique())
severity_options = sorted(filter_df['Severity'].dropna().unique())

selected_states = st.sidebar.multiselect(
    "Select States", states, default=states[:3]
)

selected_time = st.sidebar.multiselect(
    "Select Time of Day", time_options, default=time_options
)

selected_weather = st.sidebar.multiselect(
    "Select Weather Condition", weather_options, default=weather_options[:5]
)

selected_severity = st.sidebar.multiselect(
    "Select Severity Level", severity_options, default=severity_options
)

min_vis, max_vis = st.sidebar.slider(
    "Visibility Range (miles)", 0.0, 15.0, (0.0, 15.0)
)

start_date, end_date = st.sidebar.date_input(
    "Date Range",
    value=(pd.to_datetime("2019-01-01").date(),
           pd.to_datetime("2023-12-31").date())
)

start_date = datetime.combine(start_date, datetime.min.time())
end_date = datetime.combine(end_date, datetime.max.time())

@st.cache_data(ttl=600)
def load_filtered_data(
    df,
    states, timeofday, weather, severity,
    min_vis, max_vis, start_date, end_date
):
    return df[
        (df['State'].isin(states)) &
        (df['TimeOfDay'].isin(timeofday)) &
        (df['Weather_Condition'].isin(weather)) &
        (df['Severity'].isin(severity)) &
        (df['Visibility(mi)'].between(min_vis, max_vis)) &
        (df['Start_Time'].between(start_date, end_date))
    ]

if not selected_states or not selected_time or not selected_weather or not selected_severity:
    st.warning("Please select at least one value for each filter.")
    st.stop()

df = load_filtered_data(
    df_full,
    selected_states,
    selected_time,
    selected_weather,
    selected_severity,
    min_vis,
    max_vis,
    start_date,
    end_date
)

# =======================
# Feature Engineering
# =======================
df['hour'] = df['Start_Time'].dt.hour
df['weekday'] = df['Start_Time'].dt.day_name()
df['month'] = df['Start_Time'].dt.month_name()

st.sidebar.markdown(f"### Filtered records: {df.shape[0]}")

# =======================
# Tabs
# =======================
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Univariate", "Bivariate", "Multivariate", "Geospatial", "Help"]
)

# =======================
# Univariate Analysis
# =======================
with tab1:
    st.header("Univariate Analysis")

    col1, col2 = st.columns(2)

    with col1:
        hour_counts = df['hour'].value_counts().sort_index()
        fig, ax = plt.subplots()
        ax.plot(hour_counts.index, hour_counts.values, marker='o')
        ax.set_title("Accidents by Hour")
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Number of Accidents")
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots()
        sns.countplot(
            x='weekday',
            data=df,
            order=df['weekday'].value_counts().index,
            palette='Set2',
            ax=ax
        )
        ax.set_title("Accident Distribution by Weekday")
        ax.tick_params(axis='x', rotation=45)
        st.pyplot(fig)

    col3, col4 = st.columns(2)

    with col3:
        top_cities = df['City'].value_counts().head(10)
        fig, ax = plt.subplots()
        sns.barplot(x=top_cities.values, y=top_cities.index, ax=ax)
        ax.set_title("Top 10 Cities by Accident Count")
        st.pyplot(fig)

    with col4:
        month_counts = df['month'].value_counts()
        fig, ax = plt.subplots()
        ax.plot(month_counts.index, month_counts.values, marker='s', linestyle='--')
        ax.set_title("Monthly Accident Trend")
        ax.tick_params(axis='x', rotation=90)
        st.pyplot(fig)

# =======================
# Bivariate Analysis
# =======================
with tab2:
    st.header("Bivariate Analysis")

    col1, col2 = st.columns(2)

    with col1:
        weekday_counts = df['weekday'].value_counts()
        fig, ax = plt.subplots()
        ax.pie(weekday_counts, labels=weekday_counts.index, autopct='%1.1f%%')
        ax.set_title("Accident Distribution by Weekday")
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots()
        sns.violinplot(x='hour', y='Severity', data=df, inner='quartile', ax=ax)
        ax.set_title("Severity Distribution Across Hours")
        st.pyplot(fig)

    col3, col4 = st.columns(2)

    with col3:
        top_cities = df['City'].value_counts().head(10).index
        city_df = df[df['City'].isin(top_cities)]
        fig, ax = plt.subplots()
        sns.boxplot(x='City', y='Distance(mi)', data=city_df, ax=ax)
        ax.tick_params(axis='x', rotation=45)
        ax.set_title("Accident Distance Distribution by City")
        st.pyplot(fig)

    with col4:
        fig, ax = plt.subplots()
        sns.countplot(x='month', data=df, order=df['month'].value_counts().index, ax=ax)
        ax.set_title("Accidents by Month")
        ax.tick_params(axis='x', rotation=90)
        st.pyplot(fig)

# =======================
# Multivariate Analysis
# =======================
with tab3:
    st.header("Multivariate Analysis")

    col1, col2 = st.columns(2)

    with col1:
        heatmap_df = df.pivot_table(
            values='Severity',
            index='weekday',
            columns='hour',
            aggfunc='mean'
        )
        fig, ax = plt.subplots()
        sns.heatmap(heatmap_df, cmap='coolwarm', linewidths=0.5, ax=ax)
        ax.set_title("Avg Severity Heatmap (Weekday vs Hour)")
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots()
        sns.countplot(
            x='month',
            hue='TimeOfDay',
            data=df,
            order=df['month'].value_counts().index,
            ax=ax
        )
        ax.set_title("Accidents by Month & Time of Day")
        ax.tick_params(axis='x', rotation=90)
        st.pyplot(fig)

    col3, _ = st.columns(2)

    with col3:
        fig, ax = plt.subplots()
        sns.boxplot(x='weekday', y='Severity', data=df, ax=ax)
        ax.set_title("Severity Distribution by Weekday")
        ax.tick_params(axis='x', rotation=45)
        st.pyplot(fig)

# =======================
# Geospatial Analysis
# =======================
with tab4:
    st.header("Geospatial Analysis")

    geo_df = df.sample(min(5000, len(df)), random_state=42)

    fig, ax = plt.subplots(figsize=(8, 6))
    hb = ax.hexbin(
        geo_df['Start_Lng'],
        geo_df['Start_Lat'],
        gridsize=50,
        cmap='inferno'
    )
    fig.colorbar(hb, ax=ax)
    ax.set_title("Accident Hotspots")
    st.pyplot(fig)

    st.subheader("Interactive Map")

    map_data = geo_df[['Start_Lat', 'Start_Lng']].rename(
        columns={'Start_Lat': 'lat', 'Start_Lng': 'lon'}
    )
    st.map(map_data)

# =======================
# Help
# =======================
with tab5:
    st.header("Dashboard Usage Guidelines")

    st.markdown("""
    ### How to Use This Dashboard

    #### Apply Filters (Left Sidebar)
    Use the sidebar filters to customize the dataset:
    - **States**: Select one or more U.S. states
    - **Time of Day**: Day / Night (Sunrise / Sunset)
    - **Weather Condition**: Filter accidents by weather type
    - **Severity Level**: Choose accident severity levels (1–4)
    - **Visibility Range**: Adjust visibility limits (miles)
    - **Date Range**: Select the accident occurrence period

    *At least one value must be selected for each filter.*

    ---

    #### Navigate Analysis Tabs
    - **Univariate**: Distribution of accidents by hour, weekday, month, and city
    - **Bivariate**: Relationship between severity and time, distance, and months
    - **Multivariate**: Severity comparison across weekdays, hours, and time of day
    - **Geospatial**: Accident hotspots and geographic distribution

    ---

    #### Interactive Map
    - Zoom and pan to explore accident locations
    - Dense regions indicate accident hotspots

    ---

    #### Data Preview
    - View the first 100 filtered accident records
    - Useful for data validation and inspection

    ---

    ### Intended Use
    This dashboard is intended for:
    - Academic projects
    - Data analytics demonstrations
    - Road safety research
    - Exploratory Data Analysis (EDA)
    """)

st.header("Filtered Dataset Preview")
st.dataframe(df.head(100))
st.write("Shape:", df.shape)

st.success("RoadSafe Analytics Dashboard Loaded Successfully")

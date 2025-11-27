import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from streamlit_local_storage import LocalStorage


st.set_page_config(page_title="Netflix Recommendation System", layout="wide")

st.title("🎬 Netflix Recommendation System Dashboard")

# --- Local Storage Setup ---
storage = LocalStorage()

# Dataset uploader
st.sidebar.header("Upload Your Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

# Load data from Local Storage if exists
if "dataset" in storage:
    df = pd.DataFrame(storage.getItem("dataset"))
else:
    df = None

# When user uploads file
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Clean messy columns
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Handle missing values
    df = df.replace(["", " ", "NA", "N/A", "null"], np.nan)
    df = df.fillna({
        "genre": "Unknown",
        "country": "Unknown",
        "user_review": "Not Provided"
    })

    # Save to local storage
    storage.setItem("dataset", df.to_dict(orient="records"))
    st.success("Dataset loaded and stored in local storage!")

if df is None:
    st.info("Please upload a dataset to continue.")
    st.stop()

st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

# KPI Row
st.markdown("## 🔢 Key Performance Indicators")
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Titles", len(df))
col2.metric("Unique Genres", df['genre'].nunique())
col3.metric("Average Rating", round(df['rating'].mean(), 2))
col4.metric("Average Duration (min)", round(df['duration_minutes'].mean(), 2))

# Visualizations
st.markdown("## 📈 Visualizations")

# Genre Count
fig1 = px.bar(df['genre'].value_counts().reset_index(),
              x='index', y='genre',
              labels={'index': 'Genre', 'genre': 'Count'},
              title="Genre Distribution")
st.plotly_chart(fig1, use_container_width=True)

# Ratings Distribution
fig2 = px.histogram(df, x='rating', nbins=20,
                    title="Rating Distribution")
st.plotly_chart(fig2, use_container_width=True)

# Country Distribution
fig3 = px.pie(df, names='country',
              title="Content by Country")
st.plotly_chart(fig3, use_container_width=True)

# Recommendation Logic (Simple)
st.markdown("## 🎯 Content-Based Recommendations")

selected_genre = st.selectbox("Pick a Genre", df['genre'].unique())

recommended = df[df['genre'] == selected_genre].sort_values("rating", ascending=False).head(10)

st.write("### Top Recommendations:")
st.dataframe(recommended)


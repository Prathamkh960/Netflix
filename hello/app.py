import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from streamlit_local_storage import LocalStorage


st.set_page_config(page_title="Netflix Recommendation System", layout="wide")

st.title("🎬 Netflix Recommendation System Dashboard")

# --- Local Storage ---
storage = LocalStorage()

# Try loading dataset from LocalStorage
saved_data = storage.getItem("dataset")
if saved_data:
    df = pd.DataFrame(saved_data)
else:
    df = None

# Sidebar Upload Section
st.sidebar.header("Upload Your Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # -----------------
    # CLEANING THE DATA
    # -----------------

    # Normalize column names
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )

    # Replace common messy values with NaN
    df = df.replace(["", " ", "NA", "N/A", "null", "None"], np.nan)

    # Fill missing values
    df = df.fillna({
        "genre": "Unknown",
        "country": "Unknown",
        "user_review": "Not Provided"
    })

    # Store cleaned dataset in Local Storage
    storage.setItem("dataset", df.to_dict(orient="records"))
    st.success("Dataset uploaded, cleaned, and saved in local storage!")

# Stop if no data available
if df is None:
    st.info("Upload a CSV file to start.")
    st.stop()

# -----------------
# DATA PREVIEW
# -----------------
st.subheader("📊 Dataset Preview")
st.dataframe(df.head())


# -----------------
# KPI SECTION
# -----------------
st.markdown("## 🔢 Key Performance Indicators")
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Titles", len(df))
col2.metric("Unique Genres", df["genre"].nunique())
col3.metric("Average Rating", round(df["rating"].astype(float).mean(), 2))
col4.metric("Avg Duration (min)", round(df["duration_minutes"].astype(float).mean(), 2))


# -----------------
# VISUALIZATIONS
# -----------------
st.markdown("## 📈 Visualizations")

# Genre Distribution
fig1 = px.bar(
    df["genre"].value_counts().reset_index(),
    x="index",
    y="genre",
    labels={"index": "Genre", "genre": "Count"},
    title="Genre Distribution"
)
st.plotly_chart(fig1, use_container_width=True)

# Rating Histogram
fig2 = px.histogram(
    df,
    x="rating",
    nbins=20,
    title="Rating Distribution"
)
st.plotly_chart(fig2, use_container_width=True)

# Content by Country
fig3 = px.pie(
    df,
    names="country",
    title="Content by Country"
)
st.plotly_chart(fig3, use_container_width=True)


# -----------------
# SIMPLE RECOMMENDER
# -----------------
st.markdown("## 🎯 Content-Based Recommendations")

selected_genre = st.selectbox("Choose a Genre", df["genre"].unique())

recommended = (
    df[df["genre"] == selected_genre]
    .sort_values("rating", ascending=False)
    .head(10)
)

st.write("### Top Recommendations:")
st.dataframe(recommended)


import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from streamlit_local_storage import LocalStorage


st.set_page_config(page_title="Netflix Recommendation System", layout="wide")

st.title("🎬 Netflix Recommendation System Dashboard")

# --- Local Storage ---
storage = LocalStorage()

# Try loading dataset from Local Storage
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

    # Force safe types
    if "rating" in df.columns:
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    if "duration_minutes" in df.columns:
        df["duration_minutes"] = pd.to_numeric(df["duration_minutes"], errors="coerce")

    # Fill missing values
    df = df.fillna({
        "genre": "Unknown",
        "country": "Unknown",
        "user_review": "Not Provided",
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

# Avoid errors if missing columns
avg_rating = round(df["rating"].mean(), 2) if "rating" in df.columns else "N/A"
avg_duration = (
    round(df["duration_minutes"].mean(), 2)
    if "duration_minutes" in df.columns
    else "N/A"
)

col3.metric("Average Rating", avg_rating)
col4.metric("Avg Duration (min)", avg_duration)


# -----------------
# VISUALIZATIONS
# -----------------
st.markdown("## 📈 Visualizations")

# --- FIXED GENRE COUNT PLOT ---
try:
    genre_counts = df["genre"].value_counts().reset_index()
    genre_counts.columns = ["genre", "count"]

    fig1 = px.bar(
        genre_counts,
        x="genre",
        y="count",
        title="Genre Distribution"
    )
    st.plotly_chart(fig1, use_container_width=True)
except:
    st.error("Could not generate Genre Distribution plot.")


# --- RATING DISTRIBUTION ---
if "rating" in df.columns:
    fig2 = px.histogram(
        df,
        x="rating",
        nbins=20,
        title="Rating Distribution"
    )
    st.plotly_chart(fig2, use_container_width=True)


# --- COUNTRY PIE CHART ---
try:
    fig3 = px.pie(
        df,
        names="country",
        title="Content by Country"
    )
    st.plotly_chart(fig3, use_container_width=True)
except:
    st.error("Could not generate Country Distribution plot.")


# -----------------
# SIMPLE RECOMMENDER
# -----------------
st.markdown("## 🎯 Content-Based Recommendations")

if "genre" in df.columns:
    selected_genre = st.selectbox("Choose a Genre", df["genre"].unique())

    recommended = (
        df[df["genre"] == selected_genre]
        .sort_values("rating", ascending=False)
        .head(10)
    )

    st.write("### Top Recommendations:")
    st.dataframe(recommended)
else:
    st.warning("Genre column not found in dataset. Cannot generate recommendations.")

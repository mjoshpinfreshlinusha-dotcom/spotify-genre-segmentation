import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(
    page_title="Spotify Genre Segmentation",
    page_icon="🎵",
    layout="wide"
)

st.title("🎵 Spotify Songs Genre Segmentation")
st.write("K-Means Clustering based Spotify Song Analysis")

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "outputs", "kmeans_model.joblib")
SCALER_PATH = os.path.join(BASE_DIR, "outputs", "scaler.joblib")

# Load models
try:
    kmeans = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    st.success("Models loaded successfully!")
except Exception as e:
    st.error(f"Could not load models: {e}")
    st.stop()

st.header("🎶 Enter Song Features")

col1, col2, col3 = st.columns(3)

with col1:
    danceability = st.number_input(
        "Danceability",
        min_value=0.0,
        max_value=1.0,
        value=0.5
    )

with col2:
    energy = st.number_input(
        "Energy",
        min_value=0.0,
        max_value=1.0,
        value=0.5
    )

with col3:
    loudness = st.number_input(
        "Loudness",
        min_value=-60.0,
        max_value=5.0,
        value=-5.0
    )

col4, col5, col6 = st.columns(3)

with col4:
    speechiness = st.number_input(
        "Speechiness",
        min_value=0.0,
        max_value=1.0,
        value=0.1
    )

with col5:
    acousticness = st.number_input(
        "Acousticness",
        min_value=0.0,
        max_value=1.0,
        value=0.3
    )

with col6:
    instrumentalness = st.number_input(
        "Instrumentalness",
        min_value=0.0,
        max_value=1.0,
        value=0.0
    )

col7, col8, col9 = st.columns(3)

with col7:
    liveness = st.number_input(
        "Liveness",
        min_value=0.0,
        max_value=1.0,
        value=0.1
    )

with col8:
    valence = st.number_input(
        "Valence",
        min_value=0.0,
        max_value=1.0,
        value=0.5
    )

with col9:
    tempo = st.number_input(
        "Tempo",
        min_value=0.0,
        max_value=250.0,
        value=120.0
    )

if st.button("🎯 Predict Genre Cluster", use_container_width=True):

    input_data = pd.DataFrame({
        "danceability": [danceability],
        "energy": [energy],
        "loudness": [loudness],
        "speechiness": [speechiness],
        "acousticness": [acousticness],
        "instrumentalness": [instrumentalness],
        "liveness": [liveness],
        "valence": [valence],
        "tempo": [tempo]
    })

    try:
        scaled_data = scaler.transform(input_data)
        cluster = kmeans.predict(scaled_data)

        st.success(f"🎵 Song belongs to Cluster: {cluster[0]}")

    except Exception as e:
        st.error(f"Prediction error: {e}")
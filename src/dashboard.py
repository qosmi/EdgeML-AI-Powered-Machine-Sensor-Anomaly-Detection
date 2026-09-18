"""Simple EdgeML dashboard."""

import streamlit as st

from src.pipeline import analyze


st.set_page_config(
    page_title="EdgeML",
    layout="wide",
)

st.title("EdgeML")
st.subheader("AI-Powered Machine Sensor Anomaly Detection")

data_path = st.text_input(
    "Sensor dataset",
    "data/sensor_data.csv",
)

if st.button("Analyze"):
    result = analyze(data_path)

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Records",
        f"{result['records']:,}",
    )

    col2.metric(
        "Anomalies",
        f"{result['anomalies']:,}",
    )

    col3.metric(
        "Anomaly Rate",
        f"{result['anomaly_rate']:.2%}",
    )

    st.write(
        "The anomaly detector identifies observations "
        "whose sensor patterns differ from learned normal "
        "operating behavior."
    )
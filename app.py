import streamlit as st
import pandas as pd
import json
import os
import plotly.express as px
from datetime import datetime

# UI Config
st.set_page_config(page_title="MKR Quant", page_icon="🚀", layout="centered")
st.markdown("<style>#MainMenu, footer, header {visibility: hidden;}</style>", unsafe_allow_html=True)

st.title("🚀 MKR Quant AI")

@st.cache_data(ttl=60)
def get_data():
    if os.path.exists('data.json'):
        with open('data.json', 'r') as f:
            return json.load(f)
    return None

data_package = get_data()

if data_package:
    # Handle both old and new JSON formats
    if isinstance(data_package, dict) and "portfolio" in data_package:
        df = pd.DataFrame(data_package["portfolio"])
        ts = data_package["last_update"]
    else:
        df = pd.DataFrame(data_package)
        ts = "Legacy Sync"

    st.caption(f"Last Updated: {ts}")

    # --- 1. TABLE ---
    st.subheader("Optimal Weights")
    st.dataframe(
        df.style.background_gradient(subset=['Forecast'], cmap='Greens')
        .format({'Weight': '{:.1%}', 'Forecast': '+{:.1f}%', 'Profit Factor': '{:.2f}', 'Calmar': '{:.2f}'}),
        use_container_width=True
    )

    # --- 2. DONUT CHART ---
    st.subheader("Allocation Map")
    fig = px.pie(df, values='Weight', names='Asset', hole=0.4, 
                 color_discrete_sequence=px.colors.qualitative.Pastel)
    fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

else:
    st.error("Waiting for initial data. Run GitHub Action.")

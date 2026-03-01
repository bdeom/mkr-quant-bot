import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime

# Mobile-friendly config
st.set_page_config(page_title="MKR Quant", page_icon="🚀", layout="centered")

# CSS to hide Streamlit branding
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

st.title("🚀 MKR Quant AI")

# --- DATA LOADING ---
# Use a try-except and clear cache to ensure the LATEST data.json is read
@st.cache_data(ttl=60) # Only cache for 60 seconds
def load_data():
    if os.path.exists('data.json'):
        with open('data.json', 'r') as f:
            return json.load(f)
    return []

raw_data = load_data()

if raw_data:
    df = pd.DataFrame(raw_data)
    
    # 1. TABLE RENDERING
    st.subheader("Optimal Portfolio Weights")
    # Mapping columns to match backend output exactly
    st.dataframe(
        df.style.background_gradient(subset=['Forecast'], cmap='Greens')
        .format({'Weight': '{:.1%}', 'Forecast': '+{:.2f}%', 'Profit Factor': '{:.2f}', 'Calmar': '{:.2f}'}),
        use_container_width=True
    )
    
    # 2. PIE CHART RENDERING (Allocation Map)
    # Ensure columns 'Asset' and 'Weight' exist
    st.subheader("Allocation Map")
    if 'Asset' in df.columns and 'Weight' in df.columns:
        st.pie_chart(data=df, names='Asset', values='Weight')
    else:
        st.warning("Data columns missing for chart.")

    st.caption(f"Last sync: {datetime.now().strftime('%H:%M:%S UTC')}")
else:
    st.error("No data found. Please run the GitHub Action.")

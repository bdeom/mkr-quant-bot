import streamlit as st
import pandas as pd
import json
from datetime import datetime

# Mobile-friendly config
st.set_page_config(page_title="MKR Quant", page_icon="🚀", layout="centered")

# CSS to hide Streamlit branding so it looks like a native iOS app
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

st.title("🚀 MKR Quant AI")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

try:
    with open('data.json', 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    
    # Format data for mobile viewing
    st.subheader("Optimal Portfolio Weights")
    st.dataframe(
        df.style.background_gradient(subset=['Forecast'], cmap='Greens')
        .format({'Weight': '{:.1%}', 'Forecast': '+{:.1f}%', 'Calmar': '{:.2f}'}),
        use_container_width=True
    )
    
    st.subheader("Allocation Map")
    st.pie_chart(df, names="Asset", values="Weight")

except Exception as e:
    st.error("Engine is currently calculating daily weights. Please check back soon.")

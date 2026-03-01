import streamlit as st
import pandas as pd
import json
import os
import plotly.express as px

# UI Config
st.set_page_config(page_title="MKR Quant", page_icon="🚀", layout="centered")
st.markdown("<style>#MainMenu, footer, header {visibility: hidden;}</style>", unsafe_allow_html=True)

st.title("🚀 MKR Quant AI")

@st.cache_data(ttl=30)
def get_data():
    if os.path.exists('data.json'):
        with open('data.json', 'r') as f:
            return json.load(f)
    return None

data_package = get_data()

if data_package:
    # Handle JSON format
    if isinstance(data_package, dict) and "portfolio" in data_package:
        df = pd.DataFrame(data_package["portfolio"])
        ts = data_package.get("last_update", "Unknown")
    else:
        df = pd.DataFrame(data_package)
        ts = "Legacy Sync"

    st.caption(f"Last Updated: {ts}")

    # --- BULLETPROOF CHECK: Is the portfolio empty? ---
    if not df.empty:
        st.subheader("Optimal Weights")
        
        # Safely build formatting dictionary based on existing columns
        format_dict = {}
        if 'Weight' in df.columns: format_dict['Weight'] = '{:.1%}'
        if 'Forecast' in df.columns: format_dict['Forecast'] = '+{:.1f}%'
        if 'Profit Factor' in df.columns: format_dict['Profit Factor'] = '{:.2f}'
        if 'Calmar' in df.columns: format_dict['Calmar'] = '{:.2f}'

        # Safely apply color gradient only if 'Forecast' exists
        if 'Forecast' in df.columns:
            styled_df = df.style.background_gradient(subset=['Forecast'], cmap='Greens').format(format_dict)
        else:
            styled_df = df.style.format(format_dict)

        st.dataframe(styled_df, use_container_width=True)

        # --- DONUT CHART ---
        st.subheader("Allocation Map")
        if 'Asset' in df.columns and 'Weight' in df.columns:
            fig = px.pie(df, values='Weight', names='Asset', hole=0.4, 
                         color_discrete_sequence=px.colors.qualitative.Pastel)
            fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Chart cannot be rendered: Missing Asset/Weight columns.")
            
    else:
        # Graceful handling of a Bear Market (0 coins passed the MKR filter)
        st.info("🐻 **Market Protection Active:** No assets are currently in a confirmed MKR uptrend. The optimal mathematical allocation today is **100% Cash/Stablecoins**.")

else:
    st.error("Waiting for initial data. Run the GitHub Action.")

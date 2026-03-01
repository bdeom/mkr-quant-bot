import streamlit as st
import pandas as pd
import json
import os
import plotly.express as px

st.set_page_config(page_title="MKR Quant", page_icon="🚀", layout="centered")

# CSS to make it look like a native app
st.markdown("<style>#MainMenu, footer, header {visibility: hidden;}</style>", unsafe_allow_html=True)

st.title("🚀 MKR Quant AI")

if os.path.exists('data.json'):
    with open('data.json', 'r') as f:
        df = pd.DataFrame(json.load(f))
    
    if not df.empty:
        # --- 1. TABLE ---
        st.subheader("Optimal Weights")
        st.dataframe(df.style.background_gradient(subset=['Forecast'], cmap='Greens')
                     .format({'Weight': '{:.1%}', 'Forecast': '+{:.1f}%', 'Profit Factor': '{:.2f}', 'Calmar': '{:.2f}'}),
                     use_container_width=True)
        
        # --- 2. ALLOCATION MAP (Fixing the Error) ---
        st.subheader("Allocation Map")
        fig = px.pie(df, values='Weight', names='Asset', 
                     hole=0.4, 
                     color_discrete_sequence=px.colors.sequential.RdBu)
        fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Data file is empty. Run GitHub Action.")
else:
    st.error("data.json not found.")

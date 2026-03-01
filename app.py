import streamlit as st
import pandas as pd
import json
import os
import plotly.express as px

st.set_page_config(page_title="MKR Quant L/S", layout="centered")
st.markdown("<style>#MainMenu, footer, header {visibility: hidden;}</style>", unsafe_allow_html=True)

st.title("🚀 MKR Wealth: Long/Short Engine")

if os.path.exists('data.json'):
    with open('data.json', 'r') as f:
        data_package = json.load(f)
    
    df = pd.DataFrame(data_package["portfolio"])
    st.caption(f"Last Updated: {data_package['last_update']}")

    # --- Table Styling ---
    # Green for Long, Red for Short
    def color_side(val):
        color = '#27ae60' if val == 'LONG' else '#e74c3c'
        return f'color: white; background-color: {color}; font-weight: bold; border-radius: 5px;'

    st.subheader("Dual-Sided Allocations")
    st.dataframe(
        df.style.applymap(color_side, subset=['Side'])
        .format({'Weight': '{:.1%}', 'Forecast': '+{:.1f}%', 'Profit Factor': '{:.2f}', 'Calmar': '{:.2f}'}),
        use_container_width=True
    )

    # --- Donut Chart ---
    st.subheader("Portfolio Composition")
    fig = px.pie(df, values='Weight', names='Asset', hole=0.4, 
                 color='Side', color_discrete_map={'LONG':'#27ae60', 'SHORT':'#e74c3c'})
    st.plotly_chart(fig, use_container_width=True)

else:
    st.error("Engine warming up... please run GitHub Action.")

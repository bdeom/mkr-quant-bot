import streamlit as st
import pandas as pd
import json
import os
import plotly.express as px

st.set_page_config(page_title="MKR Top 8 Leaderboard", layout="centered")
st.markdown("<style>#MainMenu, footer, header {visibility: hidden;}</style>", unsafe_allow_html=True)

st.title("🏆 MKR Top 8 Leaderboard")

if os.path.exists('data.json'):
    with open('data.json', 'r') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data["portfolio"])
    st.caption(f"Strategy Pulse: {data['last_update']}")

    # Table with visual Side indicator
    def color_side(val):
        return 'background-color: #27ae60; color: white' if val == 'LONG' else 'background-color: #e74c3c; color: white'

    st.subheader("Top Ranked Opportunities")
    st.dataframe(
        df.style.applymap(color_side, subset=['Side'])
        .format({'Weight': '{:.1%}', 'Forecast': '{:.1f}%', 'Profit Factor': '{:.2f}', 'Calmar': '{:.2f}'}),
        use_container_width=True
    )

    # Visualizing the 8-way split
    st.subheader("Capital Allocation")
    fig = px.bar(df, x='Asset', y='Weight', color='Side', 
                 color_discrete_map={'LONG':'#27ae60', 'SHORT':'#e74c3c'},
                 text_auto='.1%')
    fig.update_layout(xaxis_title="Top 8 Assets", yaxis_title="Portfolio Weight")
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Scanning markets for the Top 8 MKR opportunities...")

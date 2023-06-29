import streamlit as st
from PIL import Image


st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("# Welcome to Bridges by Velvaere!")

# Center image
col1, col2, col3 = st.columns(3)
with col1:
    st.write(' ')
with col2:
    st.image("./velvaere.png")
with col3:
    st.write(' ')


st.sidebar.success("Select a page above.")

st.markdown(
    """
    Bridges predicts **postpartum depression** in new mothers from the convenience of your home.
    
    👈 Explore our pages to learn more about our product!
"""
)
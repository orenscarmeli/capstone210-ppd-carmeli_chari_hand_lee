import streamlit as st
from PIL import Image

hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       
       </style>
       """
st.set_page_config(
    page_title="Welcome to Velvaere",
    page_icon="./page_logo.png",
    # page_bg_color="0e0e35",
)

st.markdown(hide_default_format, unsafe_allow_html=True)

st.sidebar.header("Home")
st.image("./banner2.png")
st.write("")
st.write("")

col1, col2 = st.columns(2)

with col1:
    st.subheader("**What is Bridges?**")
    st.write(
        "Bridges is a web app by Velvaere offering two machine learning based prediction tools to support PPD (Postpartum Depression) diagnosis."
    )
with col2:
    st.subheader("**Why do we need it?**")
    st.write(
        "PPD is observed in 10-15% of new mothers but many cases aren't recognized. PPD diagnosis is important to ensure mother and baby safety and wellbeing as difficult temperament, poor self-regulation, and behavior problems have been observed in infants of depressed mothers. Bridges aims to improve diagnosis by using multimodal ML based approaches to prediction."
    )

st.divider()

st.subheader("**How do I use Bridges?**")
st.image("./home_how_to_use.png")
st.divider()

# st.subheader("**Who should use Bridges?**")
# st.write("New mothers who:")
# st.write("""
# - Are interested in using a novel ML based approach to predicting PPD
# - Hesitant to share with their friends/family for whatever reason

# """)
# st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("**Tell me more about PPD**")
    st.write(
        "The diagnosis criteria by the American Psychiatric Association is 2+ weeks of persistent:"
    )
    st.write("1. Depressed mood or")
    st.write(
        "2. Loss of interest in daily activities with four associated symptoms onsetting within 4 weeks after birth"
    )
    st.write(
        "Currently the Edinburgh Postnatal Depression Scale is used for screening but there is no diagnostic test for PPD."
    )
with col2:
    st.subheader("**What relevant research is out there?**")
    st.write("Link to papers with executive summary")

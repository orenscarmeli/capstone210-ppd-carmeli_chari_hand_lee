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
    page_icon="./page_logo.png"
)
st.markdown(hide_default_format, unsafe_allow_html=True)


st.sidebar.header("Home")

st.image("./banner2.png")
st.write("")
st.write("")

col1, col2 = st.columns(2)

with col1:
    st.subheader("**What is Bridges?**")
    st.write("Bridges is a web app by Velvaere offering two machine learning based prediction tools to support PPD (Postpartum Depression) diagnosis.")
with col2:
    st.subheader("**Why do we need it?**")
    st.write("PPD is observed in 10-15% of new mothers but many cases aren't recognized. PPD diagnosis is important to ensure mother and baby safety and wellbeing as difficult temperament, poor self-regulation, and behavior problems have been observed in infants of depressed mothers. Bridges aims to improve diagnosis by using multimodal ML based approaches to prediction.")

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
    st.write("The diagnosis criteria by the American Psychiatric Association is 2+ weeks of persistent:")
    st.write("1. Depressed mood or")
    st.write("2. Loss of interest in daily activities with four associated symptoms onsetting within 4 weeks after birth")
    st.write("Currently the Edinburgh Postnatal Depression Scale is used for screening but there is no diagnostic test for PPD.")
with col2:
    st.subheader("**What relevant research is out there?**")
    st.markdown("[Wilkie, E.,](https://www.cambridge.org/core/journals/european-psychiatry/article/prediction-of-postpartum-depression-and-anxiety-based-on-clinical-interviews-and-symptom-selfreports-of-depression-and-anxiety-during-pregnancy/C86C9329EAF29521377E5BF9327A9650/) Gillet, V., Talati, A., Posner, J., & Takser, L. (2022). Prediction of post-partum depression and anxiety based on clinical interviews and symptom self-reports of depression and anxiety during pregnancy. European Psychiatry, 65(S1), S268–S269.")
    st.markdown("[Bloch, M.,](https://pubmed.ncbi.nlm.nih.gov/16377359/) Rotenberg, N., Koren, D., & Klein, E. (2006b). Risk factors for early postpartum depressive symptoms. General Hospital Psychiatry, 28(1), 3–8.")
    st.markdown("[Gao, S.,](https://onlinelibrary.wiley.com/doi/10.1111/cns.13048) Calhoun, V., & Sui, J. (2018). Machine learning in major depression: From classification to treatment outcome prediction. CNS Neuroscience & Therapeutics, 24(11), 1037–1052.")




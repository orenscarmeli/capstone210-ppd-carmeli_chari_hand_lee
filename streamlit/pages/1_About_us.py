import streamlit as st

st.set_page_config(page_title="About us",
                   page_icon="./page_logo.png")

st.image("./header_about_us.png")


st.sidebar.header("About us")
st.write(
    """
    
    ### Meet our team

    The Velvaere team consists of full stack data scientists from UC Berkeley's Masters of Information and Data Science program.
    Our goal with Bridges is to motivate additional research efforts on PPD diagnosis.
    """)


st.image("./about_us_team.png")

    
st.divider()

st.subheader("Project information")
st.write("Bridges is **not** a standalone diagnostic tool. It is intended to support and facilitate physicians in postpartum depression diagnosis and offer a convenient screening method for postpartum women.")

st.image("./about_us_models.png")

col1, col2 = st.columns(2)

with col1:
    # st.write("#### Survey Model")
    st.write("The Survey Model was developed using data from the National Health and Nutrition Examination Survey (NHANES) conducted by the Center of Disease Control and Prevention (CDC).")
with col2:
    # st.write("#### Imaging Model")
    st.write("The Imaging Model was developed using images collected by Bezmaternykh et al. (Bezmaternykh D.D., Melnikov M.Y., Savelov A.A. et al. Brain Networks Connectivity in Mild to Moderate Depression: Resting State fMRI Study with Implications to Nonpharmacological Treatment. Neural Plasticity, 2021. V. 2021. № 8846097. PP. 1-15. DOI: 10.1155/2021/8846097)")


st.divider()
st.write(
    """
    ### Mission statement

    Our mission is to prioritize and support the mental health and well-being of new mothers by developing machine learning based tools. We aim to provide a safe and compassionate space for mothers to express their concerns, fears, and challenges, while fostering a sense of community and understanding. Together, we can create a world where every mother's mental health is valued and protected, enabling them to thrive as individuals and as caregivers.

    """
)
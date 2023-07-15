import streamlit as st

st.set_page_config(page_title="About us", page_icon="")

st.markdown("# About us")
st.sidebar.header("About us")
st.write(
    """
    
    ### Meet our team

    The Velvaere team consists of data scientists from UC Berkeley's Masters of Information and Data Science program.
    """)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.image('./savita.png', caption='Savita Chari')
with col2:
    st.image('./jon.png', caption='Jon Hand')
with col3:
    st.image('./oren.png', caption='Oren Carmeli')
with col4:
    st.image('./julie.png', caption='Julie Lee')
    
st.divider()
st.write(
    """
    ### Project information

    - Bridges is **not** a standalone diagnostic tool. It is intended to support and facilitate physicians in postpartum depression diagnosis and offer a convenient screening method for postpartum women.
    """
)

st.divider()
st.write(
    """
    ### Mission statement

    Our mission is to prioritize and support the mental health and well-being of new mothers by developing machine learning based tools. We aim to provide a safe and compassionate space for mothers to express their concerns, fears, and challenges, while fostering a sense of community and understanding. Together, we can create a world where every mother's mental health is valued and protected, enabling them to thrive as individuals and as caregivers.

    """
)
import streamlit as st

st.set_page_config(page_title="Privacy policy", page_icon="")

st.markdown("# Privacy policy")
st.sidebar.header("Privacy policy")
st.write(
    """
    
    ### 1. Personal information collected

    - will any data be stored?

    - data minimization - only data necessary for model predictions will be collected
    """
)

st.divider()
st.write("""
    ### 2. How personal information is used
    """)
st.divider()
st.write("""
    ### 3. Disclosure of personal information

    - shared with physician on file
    """)
st.divider()
st.write("""
    ### 4. Your rights to data
    - accessing your personal info and how it is processed

    - deleting/limiting personal information

    - correcting personal information

    - withdrawing consent to personal information
    """)
st.divider()
st.write("""Use our portal on the Contact page for any privacy related requests.""")


import streamlit as st

st.set_page_config(page_title="Privacy disclosure",
                   page_icon="./page_logo.png")

st.image("./header_privacy_policy.png")

st.sidebar.header("Privacy disclosure")
st.write(
    """
    
    ### 1. Personal information collected

    We collect the following personal information from you:
    - Contact information
    - Log in credentials (email and password)
    - Responses to the survey model
    - fMRI images for the imaging model

    """
)

st.divider()
st.write("""
    ### 2. How personal information is used

    The collected information is used for the following purposes:
    - Respond to you regarding any inquiries or questions
    - Generate predictions for Postpartum Depression 

    """)
st.divider()
st.write("""
    ### 3. Disclosure of personal information

    We use and may disclose this information to your health care provider.

    """)
st.divider()
st.write("""
    ### 4. Your rights to data
    - Accessing your personal information and knowing how it is processed

    - Deleting/limiting personal information

    - Correcting personal information

    - Withdrawing consent to personal information at any point in time
    """)
st.divider()
st.write("""Use our portal on the Contact page for any privacy related requests.""")


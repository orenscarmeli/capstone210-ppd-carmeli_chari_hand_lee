import streamlit as st

st.set_page_config(page_title="Neuroimaging Model", page_icon="")

st.markdown("# Neuroimaging Model")
st.sidebar.header("Neuroimaging Model")
st.write(
    """
    - fMRI must have been collected while subject had eyes closed
    - provide examples of acceptable/unacceptable images?
    
    """
)

st.file_uploader('Upload fMRI file (specify format):')

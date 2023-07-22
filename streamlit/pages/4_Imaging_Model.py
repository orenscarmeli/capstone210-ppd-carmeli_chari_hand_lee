import streamlit as st
import gzip
import nibabel as nb
import os

dirname = os.path.dirname(os.path.realpath('__file__'))


st.set_page_config(page_title="Neuroimaging Model", page_icon="")

st.markdown("# Neuroimaging Model")
st.sidebar.header("Neuroimaging Model")
st.write(
    """
    - fMRI must have been collected while subject had eyes closed
    - provide examples of acceptable/unacceptable images?
    
    """
)

uploaded_file = st.file_uploader('Upload fMRI file (compressed .gz or /nii):', type=['gz', 'nii'])
if uploaded_file is not None:
    file_details = {"FileName":uploaded_file.name, "FileType":uploaded_file.type}
    st.write(file_details)
    with open(os.path.join('./imageUploads', uploaded_file.name), "wb") as f: 
        f.write(uploaded_file.getbuffer())         
    st.success("Uploaded File")
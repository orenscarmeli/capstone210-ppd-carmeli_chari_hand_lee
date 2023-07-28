import streamlit as st
import os

st.set_page_config(page_title="Imaging Model", page_icon="./page_logo.png")

st.image("./header_imaging.png")

st.sidebar.header("Imaging Model")
st.write(
    """
    Upload an fMRI image to be used to predict for PPD
    - fMRI must have been collected while lying still with eyes closed 
    - Please note any head movement by 2 mm or 2 degrees could negatively impact model performance
    - Include color/file type/size specifications
    
    """
)

st.write("Include final performance metrics.")


uploaded_file = st.file_uploader(
    "Upload fMRI file (.nii compressed as .gz):", type=["gz"]
)
if uploaded_file is not None:
    file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type}
    st.write(file_details)
    with open(os.path.join("./imageUploads", uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success("Uploaded File")

st.divider()

# st.image("./imaging_good-example.png")

col1, col2, col3 = st.columns(3)

with col1:
    st.write("Example of an acceptable image:")
    st.image("./imaging_good-example.png")
with col2:
    st.write("")
with col3:
    st.write("Example of a poor image:")

import streamlit as st

st.set_page_config(page_title="Contact us", page_icon="")

st.markdown("# Contact us")
st.sidebar.header("Contact us")
st.write(
    """
    Submit a question to us using the form below.
    
    """
)

with st.form(key='contact_form'):
    name_input = st.text_input(label='Name:')
    email_input = st.text_input(label='E-mail:')
    question_input = st.text_input(label='Question:')
    submit = st.form_submit_button(label='Submit')
    st.write(
    """
    The information you provide will be treated in accordance with Velvaere's Privacy Policy.

    """)

if submit:
    st.write("Thank you,", name_input, "! We received your submission.")

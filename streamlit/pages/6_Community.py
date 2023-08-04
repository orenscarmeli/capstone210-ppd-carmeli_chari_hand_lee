import streamlit as st

st.set_page_config(page_title="Community",
                   page_icon="./page_logo.png")

st.image("./header_community.png")
st.sidebar.header("🤝 Community")

community_posts = []

with st.form("post_form"):
    initial_post = st.text_area("Share your experience or ask any questions to fellow mothers here:")
    submitted = st.form_submit_button("Share post")


if submitted:
    community_posts.append({"initial_post": initial_post, "responses": []})

if community_posts:
    st.header("Community Posts")
    for idx, post in enumerate(community_posts):
        st.subheader(f"Conversations {idx + 1}:")
        st.write(post["initial_post"])
        st.write("Responses:")
        for answer in post["responses"]:
            st.write(f"- {responses}")

with st.form("post_response_form"):
    selected_question = st.selectbox("Select a post to respond to:", [post["initial_post"] for post in community_posts])
    response = st.text_area("Post your response here:")
    submitted = st.form_submit_button("Post Response")

if submitted:
    for post in community_posts:
        if post["initial_post"] == selected_question:
            post["responses"].append(response)
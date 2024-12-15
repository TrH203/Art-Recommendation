import streamlit as st

st.set_page_config(
    page_title = "Home",
    layout="wide",  # Use "wide" layout to make the container bigger
)


# Add custom CSS to adjust the container width further
st.markdown(
    """
    <style>
    .main {
        max-width: 1200px; /* Adjust the width as needed */
        margin: auto;     /* Center the container */
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("Home")
st.sidebar.success("Welcome to Home!")
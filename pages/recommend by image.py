import streamlit as st
from PIL import Image

# Title of the app
st.title("Image Upload and Recommedation")

# Upload image
uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Open and display the uploaded image
    image = Image.open(uploaded_image)

    # Layout with two columns
    col1, col2 = st.columns(2)

    # Show the uploaded image in the first column
    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)

    # Prediction area in the second column (placeholder for now)
    with col2:
        st.write("Recommedation Area")



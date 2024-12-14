import streamlit as st

# Title of the app
st.title("Art Style and Artistic Elements Selector")

# Create a sidebar for selections
st.sidebar.header("Choose Artistic Features")

# Select art style
art_style = st.sidebar.selectbox(
    "Select Art Style",
    ["Impressionism", "Cubism", "Abstract", "Surrealism", "Realism", "Pop Art", "Art Nouveau"]
)

# Select artistic elements
artistic_elements = st.sidebar.multiselect(
    "Select Artistic Elements",
    ["Color Palette", "Composition", "Brush Strokes", "Textures", "Shading", "Perspective"]
)

# Select digital art options
digital_art = st.sidebar.radio(
    "Choose Digital Art Style",
    ["Vector Art", "Digital Painting", "Pixel Art", "3D Rendering"]
)

# Select sculpture type
sculpture_type = st.sidebar.selectbox(
    "Select Sculpture Type",
    ["Classical", "Modern", "Abstract", "Mixed Media", "Kinetic"]
)

# st.write(f"### Art Style: {art_style}")
# st.write(f"### Artistic Elements: {', '.join(artistic_elements)}")
# st.write(f"### Digital Art Style: {digital_art}")
# st.write(f"### Sculpture Type: {sculpture_type}")

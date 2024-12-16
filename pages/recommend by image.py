import streamlit as st
from PIL import Image, UnidentifiedImageError
import torch
from torchvision import models, transforms
from torch import nn
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import json

# Device configuration
device = "cpu"

# Load pre-trained model and create feature extractor
@st.cache_data
def load_model():
    vgg19_model = models.vgg19(pretrained=False)
    model_path = 'model.pth'
    state_dict = torch.load(model_path, map_location=device)
    vgg19_model.classifier[6] = nn.Linear(vgg19_model.classifier[6].in_features, 5)
    vgg19_model.load_state_dict(state_dict)
    vgg19_model.eval()
    feature_extractor = nn.Sequential(*list(vgg19_model.features.children())).to(device)
    return feature_extractor

feature_extractor = load_model()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Extract features from an image
def extract_features(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
    except UnidentifiedImageError:
        return None
    with torch.no_grad():
        features = feature_extractor(img_tensor)
    return features.flatten().cpu().numpy()

# Load image paths and features
@st.cache_data
def load_data():
    with open('image_paths.pkl', 'rb') as file:
        image_paths = pickle.load(file)
    image_features = np.load("features.npy")
    return image_paths, image_features

image_paths, image_features = load_data()

# Find similar images
@st.cache_data
def find_similar_images(query_image_path, top_k=5):
    query_features = extract_features(query_image_path)
    if query_features is None:
        return []
    similarities = cosine_similarity([query_features], image_features)[0]
    sorted_indices = np.argsort(similarities)[::-1][1:top_k]  # Skip the query image
    return [(image_paths[i], similarities[i]) for i in sorted_indices]

# Streamlit UI
st.title("Image Upload and Recommendation")

# Feedback file setup
feedback_file = "user_data.json"
if not os.path.exists(feedback_file):
    with open(feedback_file, "w") as f:
        json.dump({"like": [], "dislike": []}, f)

# Load feedback
if "like" not in st.session_state:
    with open(feedback_file, "r") as f:
        feedback_data = json.load(f)
    st.session_state["like"] = feedback_data.get("like", [])
    st.session_state["dislike"] = feedback_data.get("dislike", [])

# Image upload widget
uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Display uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Save uploaded image for processing
    query_image_path = "temp_uploaded_image.jpg"
    image.save(query_image_path)

    # Find similar images
    find_similar_images.clear()
    similar_images = find_similar_images(query_image_path, top_k=51)

    # Display similar images
    st.markdown("### Similar Images")
    images_per_row = 5

    for i in range(0, len(similar_images), images_per_row):
        cols = st.columns(images_per_row)
        for col, (similar_image, sim_score) in zip(cols, similar_images[i:i + images_per_row]):
            with col:
                st.image(similar_image, caption=f"Sim: {sim_score:.4f}", use_container_width=True)

                reacts = st.columns(2)
                
                # Like/Dislike buttons
                
                with reacts[0]:
                    if st.button("👍", key=f"like_{similar_image}"):
                        if similar_image not in st.session_state["like"]:
                            st.session_state["like"].append(similar_image)
                        if similar_image in st.session_state["dislike"]:
                            st.session_state["dislike"].remove(similar_image)
                with reacts[1]:
                    if st.button("👎", key=f"dislike_{similar_image}"):
                        if similar_image not in st.session_state["dislike"]:
                            st.session_state["dislike"].append(similar_image)
                        if similar_image in st.session_state["like"]:
                            st.session_state["like"].remove(similar_image)

    # Save feedback
    with open(feedback_file, "w") as f:
        json.dump({"like": st.session_state["like"], "dislike": st.session_state["dislike"]}, f)

    # Feedback summary
    st.markdown("### Feedback Summary")
    st.json({"like": st.session_state["like"], "dislike": st.session_state["dislike"]})

import streamlit as st
from PIL import Image
import torch
from torchvision import models, transforms
from torch import nn
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
from PIL import UnidentifiedImageError

# Title of the app
st.title("Recommend by Like and Dislike")

# Set device to CPU or GPU
device = "cpu"


# Function to load the model and feature extractor once
def load_model_and_feature_extractor(model_path, device):
    # Define VGG19 model architecture
    vgg19_model = models.vgg19(pretrained=False)

    # Load the model state_dict (assuming model.pth is the trained model file)
    state_dict = torch.load(model_path, map_location=device)

    # Override last layers
    num_classes = 5
    vgg19_model.classifier[6] = nn.Linear(vgg19_model.classifier[6].in_features, num_classes)
    vgg19_model = vgg19_model.to(device=device)

    # Load model weights
    vgg19_model.load_state_dict(state_dict)
    vgg19_model.eval()  # Set model to evaluation mode

    # Create a feature extractor from the model's features
    feature_extractor = nn.Sequential(*list(vgg19_model.features.children())).to(device)

    return vgg19_model, feature_extractor


# Function to extract features from an image
def extract_features(image_path, feature_extractor, device="cpu"):
    try:
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)  # Add batch dimension and move to device
    except UnidentifiedImageError:
        return None
    with torch.no_grad():
        features = feature_extractor(img_tensor)
    return features.flatten().cpu().numpy()  # Return features on CPU for further processing


# Function to load user preferences from the JSON file
def load_user_preferences(file_path="user_preferences.json"):
    try:
        with open(file_path, 'r') as f:
            user_preferences = json.load(f)
        return user_preferences
    except FileNotFoundError:
        st.error(f"File {file_path} not found.")
        return {}


# Function to find similar images based on cosine similarity
def find_similar_images(query_features, image_features, image_paths, liked_images, disliked_images, top_k=5):
    similarities = cosine_similarity([query_features], image_features)[0]
    sorted_indices = np.argsort(similarities)[::-1]  # Sort in descending order

    # Filter out images that are already liked or disliked
    filtered_indices = [
        i for i in sorted_indices
        if image_paths[i] not in liked_images and image_paths[i] not in disliked_images
    ]

    # Get the top_k similar images
    top_similar_images = filtered_indices[:top_k]

    return [(image_paths[i], similarities[i]) for i in top_similar_images]


# Load model and feature extractor once, if not already loaded
# Load model and feature extractor once, if not already loaded
if "vgg19_model" not in st.session_state:
    st.session_state.vgg19_model, st.session_state.feature_extractor = load_model_and_feature_extractor(
        'pages/model.pth', device)
else:
    vgg19_model = st.session_state.vgg19_model
    feature_extractor = st.session_state.feature_extractor

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load image paths and features from pickle and numpy files
with open('pages/image_paths.pkl', 'rb') as file:
    image_paths = pickle.load(file)

image_features = np.load("pages/features.npy")

# Load user preferences
user_preferences = load_user_preferences("user_preferences.json")


# Accessing the 'default' user preferences
user = "default"
if user in user_preferences:
    liked_images = user_preferences[user]["like"]
    disliked_images = user_preferences[user]["dislike"]

    features = []  # Store the features of all liked images
    for liked_image in liked_images:
        feature = extract_features(liked_image, st.session_state.feature_extractor, device)
        if feature is not None:
            features.append(feature)

    # Calculate the average feature vector across all liked images
    if features:
        avg_features = np.mean(features, axis=0)  # Average across the feature vectors
    else:
        avg_features = None

    # Now calculate similar images based on the average features
    if avg_features is not None:
        st.markdown("Recommended Similar Images Based on Your Liked Images")

        similar_images = find_similar_images(avg_features, image_features, image_paths, liked_images, disliked_images,
                                             top_k=50)

        # Display the similar images
        for similar_image, sim_score in similar_images[1:]:
            st.image(similar_image, caption=f"Similarity Score: {sim_score:.4f}", use_container_width=False)

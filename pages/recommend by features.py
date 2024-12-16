import streamlit as st
from PIL import Image, UnidentifiedImageError
import torch
from torchvision import models, transforms
from torch import nn
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
from loguru import logger

# Title of the app
st.title("Recommend by Like and Dislike")

# Set device to CPU or GPU
device = "cpu"

# Function to load the model and feature extractor once
@st.cache_resource
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
    except (UnidentifiedImageError, FileNotFoundError):
        return None
    with torch.no_grad():
        features = feature_extractor(img_tensor)
    return features.flatten().cpu().numpy()  # Return features on CPU for further processing

# Function to load user preferences from the JSON file
def load_user_preferences(file_path="user_data.json"):
    try:
        with open(file_path, 'r') as f:
            user_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        user_data = {"default": {"like": [], "dislike": []}}
    return user_data

# Function to save user preferences to the JSON file
def save_user_preferences(user_data, file_path="user_data.json"):
    with open(file_path, "w") as f:
        json.dump(user_data, f)

# Improved recommendation mechanism
def advanced_recommendation(query_features, image_features, image_paths, liked_images, disliked_images, top_k=20):
    """
    Advanced recommendation function that considers both liked and disliked images
    
    Args:
    - query_features: Feature vector of the query
    - image_features: All image features
    - image_paths: Corresponding image paths
    - liked_images: List of liked image paths
    - disliked_images: List of disliked image paths
    - top_k: Number of recommendations to return
    
    Returns:
    - List of recommended image paths with their similarity scores
    """
    # Compute cosine similarities
    similarities = cosine_similarity([query_features], image_features)[0]
    
    # Create a score modifier based on like/dislike status
    score_modifiers = np.ones_like(similarities)
    
    # Strong penalty for disliked images
    for disliked_image in disliked_images:
        if disliked_image in image_paths:
            dislike_index = image_paths.index(disliked_image)
            # Significantly reduce the similarity score for disliked images
            score_modifiers[dislike_index] = -0.5
    
    # Slight boost for images similar to liked images
    for liked_image in liked_images:
        if liked_image in image_paths:
            like_index = image_paths.index(liked_image)
            # Slightly increase the similarity score for liked images
            score_modifiers[like_index] *= 1.2
    
    # Apply the score modifiers
    modified_similarities = similarities * score_modifiers
    
    # Sort the modified similarities in descending order
    sorted_indices = np.argsort(modified_similarities)[::-1]
    
    # Filter out previously liked or disliked images
    filtered_indices = [
        i for i in sorted_indices
        if (image_paths[i] not in liked_images and 
            image_paths[i] not in disliked_images)
    ]
    
    # Get the top_k similar images
    top_similar_images = filtered_indices[:top_k]
    
    return [(image_paths[i], modified_similarities[i]) for i in top_similar_images]

# Load model and feature extractor once
vgg19_model, feature_extractor = load_model_and_feature_extractor("model.pth", device)

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load image paths and features from pickle and numpy files
with open("image_paths.pkl", "rb") as file:
    image_paths = pickle.load(file)

image_features = np.load("features.npy")

# Load user preferences
user_data = load_user_preferences("user_data.json")
user = "default"

# Initialize session state for likes and dislikes
if "like" not in st.session_state:
    st.session_state["like"] = user_data[user]["like"]
    st.session_state["dislike"] = user_data[user]["dislike"]

# Extract features for liked and disliked images
def get_aggregated_features(image_list):
    features = []
    for image in image_list:
        feature = extract_features(image, feature_extractor, device)
        if feature is not None:
            features.append(feature)
    return np.mean(features, axis=0) if features else None

# Compute aggregated features
avg_like_features = get_aggregated_features(st.session_state["like"])
avg_dislike_features = get_aggregated_features(st.session_state["dislike"])

# Recommendation mechanism
if avg_like_features is not None and avg_dislike_features is not None:
    st.markdown("### Recommended Images Based on Your Preferences:")
    
    # Create a combined query feature that emphasizes liked features 
    # and minimizes disliked features
    combined_query_feature = (
        avg_like_features * 1.5 - avg_dislike_features * 0.5
    )
    
    # Get recommendations
    similar_images = advanced_recommendation(
        combined_query_feature, 
        image_features, 
        image_paths, 
        st.session_state["like"], 
        st.session_state["dislike"], 
        top_k=100
    )
    
    # Display recommendations
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
                            
else:
    st.warning("You haven't liked or disliked any images yet. Start interacting to see recommendations!")

# Save feedback
with open("user_data.json", "w") as f:
    json.dump({
        "default": {
            "like": st.session_state["like"], 
            "dislike": st.session_state["dislike"]
        }
    }, f)

# Feedback summary
st.markdown("### Feedback Summary")
st.json({
    "default": { 
        "like": st.session_state["like"], 
        "dislike": st.session_state["dislike"]
    }
})
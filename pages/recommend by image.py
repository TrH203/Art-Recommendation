import streamlit as st
from PIL import Image
import torch
from torchvision import models, transforms
from torch import nn
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from PIL import UnidentifiedImageError
import json

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

# Function to find similar images based on cosine similarity
def find_similar_images(query_image_path, image_features, image_paths, feature_extractor, device, top_k=5):
    query_features = extract_features(query_image_path, feature_extractor, device)
    if query_features is None:
        return []

    similarities = cosine_similarity([query_features], image_features)[0]
    sorted_indices = np.argsort(similarities)[::-1][1:top_k]  # Top-k similar images
    return [(image_paths[i], similarities[i]) for i in sorted_indices]

# Function to update user preferences in a JSON file
def update_user_preferences(like_list, dislike_list, user="default"):
    preferences_file = "user_preferences.json"

    try:
        with open(preferences_file, 'r') as f:
            user_preferences = json.load(f)
    except FileNotFoundError:
        user_preferences = {}

    if user not in user_preferences:
        user_preferences[user] = {"like": [], "dislike": []}

    user_preferences[user]["like"] = list(set(user_preferences[user]["like"] + like_list))
    user_preferences[user]["dislike"] = list(set(user_preferences[user]["dislike"] + dislike_list))

    with open(preferences_file, 'w') as f:
        json.dump(user_preferences, f, indent=4)

# Check if the model is already loaded in session state
if "vgg19_model" not in st.session_state:
    # If not loaded, load the model and feature extractor
    st.session_state.vgg19_model, st.session_state.feature_extractor = load_model_and_feature_extractor('pages/model.pth', device)
else:
    # If loaded, use the session state model
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

# Streamlit UI
st.title("Image Upload and Recommendation")

# Image upload widget
uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Display uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Save uploaded image for processing
    query_image_path = "temp_uploaded_image.jpg"
    image.save(query_image_path)

    # Find the top-k similar images
    similar_images = find_similar_images(query_image_path, image_features, image_paths, feature_extractor, device, top_k=50)

    # Display similar images in a grid layout
    st.markdown("### Similar Images")

    # Define the number of images per row
    images_per_row = 5

    # Lists to store liked and disliked images
    liked_images = []
    disliked_images = []

    # Create rows dynamically
    for i in range(0, len(similar_images[1:]), images_per_row):
        cols = st.columns(images_per_row)  # Create columns for the row
        for col, (similar_image, sim_score) in zip(cols, similar_images[1 + i:1 + i + images_per_row]):
            with col:
                # Display the image with similarity score
                st.image(similar_image, caption=f"Sim: {sim_score:.4f}", use_container_width=False)

                # Create Like and Dislike buttons for each image
                like_button = st.button(f"Like", key=f"like_{similar_image}")
                dislike_button = st.button(f"Dislike", key=f"dislike_{similar_image}")

                # When the like button is clicked, add to the liked images list
                if like_button:
                    liked_images.append(similar_image)
                    st.success(f"You liked")

                # When the dislike button is clicked, add to the disliked images list
                if dislike_button:
                    disliked_images.append(similar_image)
                    st.error(f"You disliked")

    # After the user has finished, update the preferences JSON file
    if liked_images or disliked_images:
        update_user_preferences(like_list=liked_images, dislike_list=disliked_images)

        # Display a message after submitting feedback
        st.markdown("### Your feedback has been recorded.")
        st.write(f"Liked images: {liked_images}")
        st.write(f"Disliked images: {disliked_images}")

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
import io
from tqdm import tqdm
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


def remove_duplicates(similar, feature_extractor, threshold=1e-6):
    image_features = []
    image_paths = []
    sim_scores = []
    for i in similar:
        a, sim_score = i
        feature = extract_features(a, feature_extractor, device="cpu")  # Ensure it's compatible with your device
        if feature is not None:  # Only add valid features
            image_features.append(feature)
            image_paths.append(a)
            sim_scores.append(sim_score)

    image_features = np.array(image_features)
    dup = {}
    num_images = len(image_paths)
    for i in tqdm(range(num_images)):
        if image_paths[i] in dup:  # Skip if already flagged as duplicate
            continue

        # Compare current feature with the rest
        differences = np.sum(np.abs(image_features[i] - image_features[i + 1:]), axis=1)
        duplicates = np.where(differences <= threshold)[0]

        for idx in duplicates:
            dup[image_paths[i + 1 + idx]] = True

    # Filter unique paths and features
    new_paths = [path for path in image_paths if path not in dup]
    new_image_features = [image_features[i] for i, path in enumerate(image_paths) if path not in dup]
    new_sim_score = [sim_scores[i] for i, path in enumerate(image_paths) if path not in dup]

    results = []
    for i, j, z in zip(new_paths, new_image_features, new_sim_score):
        results.append((i, z))  # Returning only paths and similarity scores
    return results
# Extract features from an image
def extract_features(image_path, feature_extractor, device="cpu"):
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
    query_features = extract_features(query_image_path,feature_extractor)
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
        json.dump({"default": {"like": [], "dislike": []}}, f)

# Load feedback
if "like" not in st.session_state:
    with open(feedback_file, "r") as f:
        feedback_data = json.load(f)
    st.session_state["like"] = feedback_data.get("default", []).get("like")
    st.session_state["dislike"] = feedback_data.get("default", []).get("dislike")

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
    similar_images = remove_duplicates(similar_images, feature_extractor, threshold=1e-6)
    # Display similar images
    st.markdown("### Similar Images")
    images_per_row = 5

    for i in range(0, len(similar_images), images_per_row):
        cols = st.columns(images_per_row)
        for col, (similar_image, sim_score) in zip(cols, similar_images[i:i + images_per_row]):
            with col:
                st.image(similar_image, caption=f"Sim: {sim_score:.4f}", use_container_width=True)

                reacts = st.columns(3)
                
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
                with reacts[2]:
                    # Convert image to BytesIO stream for downloading
                    image_download = Image.open(similar_image)
                    img_byte_arr = io.BytesIO()
                    image_download.save(img_byte_arr, format='PNG')
                    img_byte_arr = img_byte_arr.getvalue()

                    st.download_button(
                        label="📥",
                        data=img_byte_arr,
                        file_name=similar_image.split("/")[-1],  # Use image name as file name
                        mime="image/png"
                    )
    # Save feedback
    with open(feedback_file, "w") as f:
        json.dump({"default": {"like": st.session_state["like"], "dislike": st.session_state["dislike"]}}, f)

    # Feedback summary
    # st.markdown("### Feedback Summary")
    # st.json({"default":{ "like": st.session_state["like"], "dislike": st.session_state["dislike"]}})
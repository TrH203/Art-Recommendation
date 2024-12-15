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

# Set device to CPU or GPU
device = "cpu"

# Define VGG19 model architecture
vgg19_model = models.vgg19(pretrained=False)

# Load the model state_dict (assuming model.pth is the trained model file)
model_path = 'model.pth'
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

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

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

# Function to remove duplicate images based on feature similarity
def remove_duplicates(similar_images, feature_extractor, threshold=1e-6):
    image_features = []
    image_paths = []
    sim_scores = []

    for image_path, sim_score in similar_images:
        feature = extract_features(image_path, feature_extractor, device="cpu")
        if feature is not None:
            image_features.append(feature)
            image_paths.append(image_path)
            sim_scores.append(sim_score)

    image_features = np.array(image_features)
    dup = {}
    num_images = len(image_paths)

    for i in tqdm(range(num_images)):
        if image_paths[i] in dup:
            continue
        differences = np.sum(np.abs(image_features[i] - image_features[i + 1:]), axis=1)
        duplicates = np.where(differences <= threshold)[0]
        for idx in duplicates:
            dup[image_paths[i + 1 + idx]] = True

    # Filter unique paths and features
    new_paths = [path for path in image_paths if path not in dup]
    new_image_features = [image_features[i] for i, path in enumerate(image_paths) if path not in dup]
    new_sim_score = [sim_scores[i] for i, path in enumerate(image_paths) if path not in dup]

    return [(path, score) for path, score in zip(new_paths, new_sim_score)]

# Function to find similar images based on cosine similarity
def find_similar_images(query_image_path, image_features, image_paths, feature_extractor, device, top_k=5):
    query_features = extract_features(query_image_path, feature_extractor, device)
    if query_features is None:
        return []

    similarities = cosine_similarity([query_features], image_features)[0]
    sorted_indices = np.argsort(similarities)[::-1][1:top_k]  # Top-k similar images
    return [(image_paths[i], similarities[i]) for i in sorted_indices]

# Load image paths and features from pickle and numpy files
with open('image_paths.pkl', 'rb') as file:
    image_paths = pickle.load(file)

image_features = np.load("features.npy")

# Streamlit UI
st.title("Image Upload and Recommendation")




# Image upload widget
uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Display uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_container_width=1000)

    # Save uploaded image for processing
    query_image_path = "temp_uploaded_image.jpg"
    image.save(query_image_path)

    # Assume these functions and variables are defined elsewhere
    # Find the top-k similar images
    similar_images = find_similar_images(query_image_path, image_features, image_paths, feature_extractor, device, top_k=50)
    similar_images = remove_duplicates(similar_images, feature_extractor)

    # Display similar images in a grid layout
    st.markdown("### Similar Images")

    # Define the number of images per row
    images_per_row = 5  # Adjust this value based on the desired layout

    # Create rows dynamically
    for i in range(0, len(similar_images[1:]), images_per_row):  # Skip the first image (itself)
        cols = st.columns(images_per_row)  # Create columns for the row
        for col, (similar_image, sim_score) in zip(cols, similar_images[1 + i:1 + i + images_per_row]):
            with col:
                st.image(similar_image, caption=f"Sim: {sim_score:.4f}", use_container_width=False)
import streamlit as st
import plotly.express as px
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
from tqdm import tqdm
from torchvision import transforms
from PIL import UnidentifiedImageError
from sklearn.metrics.pairwise import cosine_similarity
import torch
from torchvision import models
from torch.nn import Sequential
from torch import nn
from loguru import logger

logger.info("Running app...")
# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])



def remove_duplicates(similar, feature_extractor,threshold=1e-6):
    image_features = []
    image_paths = []
    sim_scores = []
    for i in similar:
        a, sim_score = i
        feature = extract_features(a,feature_extractor, device="mps")
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
        differences = np.sum(np.abs(image_features[i] - image_features[i+1:]), axis=1)
        duplicates = np.where(differences <= threshold)[0]

        for idx in duplicates:
            dup[image_paths[i+1+idx]] = True 

    # Filter unique paths and features
    new_paths = [path for path in image_paths if path not in dup]
    new_image_features = [image_features[i] for i, path in enumerate(image_paths) if path not in dup]
    new_sim_score = [sim_scores[i] for i, path in enumerate(image_paths) if path not in dup]

    results = []
    for i,j,z in zip(new_paths, new_image_features, new_sim_score):
        results.append((i,j,z))
    return results

# Feature extraction function
def extract_features(image_path, feature_extractor, device="cpu"):
    img_tensor = None
    try:
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)  # Add batch dimension and move to device
    except UnidentifiedImageError as e:
        # print(f"Image error: ",image_path)
        return None
    with torch.no_grad():
        features = feature_extractor(img_tensor)
    return features.flatten().cpu().numpy()  # Return features on CPU for further processing

# Function to find similar images
def find_similar_images(query_image_path, image_features, image_paths, feature_extractor, device, top_k=5):
    query_features = extract_features(query_image_path, feature_extractor, device)
    similarities = cosine_similarity([query_features], image_features)[0]
    sorted_indices = np.argsort(similarities)[::-1][1:top_k]  # Top-k similar images
    return [(image_paths[i], similarities[i]) for i in sorted_indices]


def plot_image(image_path):
    # Open the image
    img = Image.open(image_path)
    
    # Plot the image
    plt.imshow(img)
    plt.axis('off')  # Hide axes for a cleaner display
    plt.show()

def kmeans(data, k, max_iters=100, tol=1e-4):

    data = torch.tensor(data)
    n_samples, n_features = data.size()
    
    centers = data[torch.randperm(n_samples)[:k]]

    for i in range(max_iters):
        distances = torch.cdist(data, centers)
        labels = torch.argmin(distances, dim=1)

        # Update cluster centers
        new_centers = torch.stack([data[labels == j].mean(dim=0) for j in range(k)])

        # Check for convergence
        if torch.norm(new_centers - centers) < tol:
            break

        centers = new_centers

    return centers, labels


def plot_clusters_high_dim(data, labels, centers, n_components=2, title="K-Means Clustering"):
    
    data = torch.tensor(data)
    # Convert tensors to numpy
    data = data.cpu().numpy()
    labels = labels.cpu().numpy()
    centers = centers.cpu().numpy()

    # Apply PCA
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(data)
    reduced_centers = pca.transform(centers)

    # Plot the reduced data
    if n_components == 2:
        for i in range(len(centers)):
            cluster_points = reduced_data[labels == i]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {i+1}")
        plt.scatter(reduced_centers[:, 0], reduced_centers[:, 1], c='red', marker='x', s=200, label='Centers')
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()
    elif n_components == 3:
        # Create a DataFrame for visualization
        df = pd.DataFrame(reduced_data, columns=["PC1", "PC2", "PC3"])
        df['Cluster'] = labels
        custom_colors = ['red']  # Add as many colors as clusters
        # Streamlit app
        st.title("Interactive 3D PCA Visualization")
        fig = px.scatter_3d(df, x="PC1", y="PC2", z="PC3",
                            color="Cluster", 
                            color_discrete_sequence=custom_colors)
        fig.update_layout(width=1280, height=720)  # Set desired figure size

        st.plotly_chart(fig)

        
    else:
        raise ValueError("n_components must be 2 or 3 for plotting!")
    
    
    
if __name__ == "__main__":
    logger.info("Loading model...")
    device = "mps"
    # Define the VGG19 model architecture
    vgg19_model = models.vgg19(pretrained=False)

    # Load the model state_dict onto MPS device
    model_path = 'model.pth'
    state_dict = torch.load(model_path, map_location=device)

        # override last layers
    num_classes = 5
    vgg19_model.classifier[6] = nn.Linear(vgg19_model.classifier[6].in_features, num_classes)
    vgg19_model = vgg19_model.to(device=device)

    vgg19_model.load_state_dict(state_dict)
    vgg19_model.eval()  # Set model to evaluation mode

    logger.info("Getting feature extractor")
    # Create a feature extractor from the model's features
    feature_extractor = Sequential(*list(vgg19_model.features.children())).to(device) 
    
    
    logger.info("Loading features")
    # np.save("features.npy", image_features) # ! Danger Watch out
    image_features = np.load("features.npy")
    print("Image features: ", image_features.shape)
    
    
    dataset_path = 'art-images-drawings-painting-sculpture-engraving'

    image_paths = []

    # Function to extract features for a dataset
    for root, _, files in os.walk(dataset_path):  # Use os.walk for recursive traversal
        print("Checking ",root)
        for img_name in files:
            img_path = os.path.join(root, img_name)

            image_paths.append(img_path)
    
    logger.info("Clustering...")
    k = 5
    centers, labels = kmeans(image_features, k)
    # plot_3d_clusters(data, labels, centers)
    logger.info("Plotting...")
    plot_clusters_high_dim(image_features,labels, centers,n_components=3)
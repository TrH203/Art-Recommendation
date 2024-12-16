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
import pickle

logger.info("Running app...")
# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

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

def plot_image(image_path):
    # Open the image
    img = Image.open(image_path)
    
    # Plot the image
    plt.imshow(img)
    plt.axis('off')  # Hide axes for a cleaner display
    plt.show()

def get_sample(data,labels, ammount = 0.05):
    # Calculate 10% of the array size
    sample_size = int(len(data) * ammount)

    # Randomly select indices without replacement
    random_indices = np.random.choice(len(data), size=sample_size, replace=False)

    # Select the elements corresponding to the random indices
    random_sample_data = data[random_indices]
    random_sample_label = labels[random_indices]
    return random_sample_data, random_sample_label
    
def kmeans(data, k, max_iters=300, tol=1e-5):

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
    
    data, labels= get_sample(image_features,labels,0.2)
    
    
    data = torch.tensor(data)
    # Convert tensors to numpy
    data = data.cpu().numpy()
    labels = labels.cpu().numpy()
    centers = centers.cpu().numpy()

    # Apply PCA
    data = data[~np.isnan(data).any(axis=1)]

    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(data)
    reduced_centers = pca.transform(centers)

    fig, ax = plt.subplots(figsize=(10, 7))  # Create a Matplotlib figure

    for i in range(len(centers)): 
        cluster_points = reduced_data[labels == i]
        ax.scatter(
            cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {i + 1}"
        )

    # Plot centers
    ax.scatter(
        reduced_centers[:, 0],
        reduced_centers[:, 1],
        c="red",
        marker="x",
        s=200,
        label="Centers",
    )

    # Add labels, title, legend, and grid
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_title("2D PCA Scatter Plot")
    ax.legend()
    ax.grid(True)

    # Display the plot in Streamlit
    st.pyplot(fig)
    
    # Create a DataFrame for visualization
    df = pd.DataFrame(reduced_data, columns=["PC1", "PC2", "PC3"])
    df['Cluster'] = labels
    color_map = {
        0: "red",
        1: "blue",
        2: "green",
        3: "yello",
        4: "black",
    }
    # Streamlit app
    st.title("Interactive 3D PCA Visualization")
    fig = px.scatter_3d(df, x="PC1", y="PC2", z="PC3",
                        color="Cluster", color_discrete_map=color_map)
    
    fig.update_layout(width=1280, height=720)  # Set desired figure size

    st.plotly_chart(fig)

    
    # else:
    #     raise ValueError("n_components must be 2 or 3 for plotting!")
    
    
logger.info("Loading features")
image_features = np.load("features.npy")
print("Image features: ", image_features.shape)


dataset_path = 'art-images-drawings-painting-sculpture-engraving'

image_paths = []

with open("image_paths.pkl", "rb") as f:
    image_paths = pickle.load(f)


logger.info("Clustering...")
k = 5
centers, labels = kmeans(image_features, k)
# plot_3d_clusters(data, labels, centers)
logger.info("Plotting...")
plot_clusters_high_dim(image_features,labels, centers,n_components=3)


k = 10
centers, labels = kmeans(image_features, k)

plot_clusters_high_dim(image_features,labels, centers,n_components=3)

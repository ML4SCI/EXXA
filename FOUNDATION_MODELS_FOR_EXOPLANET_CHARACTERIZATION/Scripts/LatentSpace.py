import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, SpectralClustering
import matplotlib.pyplot as plt
import seaborn as sns
import umap
import umap.umap_ as umap

class LatentSpaceAnalyzer:
    def __init__(self, model, dataloader, model_type='vitmae'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model
        self.dataloader = dataloader
        self.model_type = model_type
        self.latents = None
        self.images=None
        self.reduced_feats = None
        self.cluster_labels = None

    def generate_latent_space_representations(self):
        self.model.to(self.device).eval()
        self.latents = []
        self.images = []
        self.image_paths = []

        for data in tqdm(self.dataloader):
            img_data = data['img_data'].to(self.device)
            img_path = data['img_path']
            p = data['img_path']
            self.image_paths.extend(img_path)
            with torch.no_grad():
                if (self.model_type == 'vitmae'):
                    latent, _, _ = model.forward_encoder(img_data, p, 0.0)
                    latent = latent[:, :1, :] # take cls token
                    
                    latent = latent.squeeze(1).cpu().numpy()
                    #print(latent.shape)
                else:
                    _, latent = model(img_data)
                    latent = latent.cpu().numpy()
                    
                self.latents.append(latent)
                if (self.model_type == 'cae'):
                    img = data['plot_img']
    
                else:
                    img = data['img_data'].cpu().numpy()
                self.images.append(img)
        self.latents = np.vstack(self.latents)
        self.images = np.vstack(self.images)
        return self.latents, self.images

    # check variance for pca and reduce_dim
    def reduce_dim(self, method = 'umap', n_components=5, random_state=42):
        if (len(self.latents) == 0):
            print("Run generate_latent_space_representations() first!!")
            return 
        if method == 'tsne':
            reducer = TSNE(n_components=n_components, random_state=random_state)
            self.reduced_feats = reducer.fit_transform(self.latents)
        elif method == 'pca':
            reducer = PCA(n_components=n_components, random_state=random_state)
            self.reduced_feats = reducer.fit_transform(self.latents)
        elif method == 'umap':
            reducer = umap.UMAP(n_components=n_components, random_state=random_state)
            self.reduced_feats = reducer.fit_transform(self.latents)

        return self.reduced_feats

    def cluster_images(self, method='spectral', n_clusters=3, random_state=42):
        self.clustering_method = method
        if (method=='spectral'):
            clusterer = SpectralClustering(n_clusters=n_clusters, assign_labels='discretize', 
                                           affinity='nearest_neighbors', 
                                           random_state=random_state)
        if (method=='kmeans'):
            clusterer = KMeans(n_clusters=n_clusters, random_state=random_state)
            
        self.cluster_labels = clusterer.fit_predict(self.reduced_feats)
        self.n_clusters = len(np.unique(self.cluster_labels))

        return self.cluster_labels

    def visualize_clusters(self, figsize=(12, 10)):
        if (self.cluster_labels is None):
            print("Run cluster_images() first!!")
            return 
        if (self.reduced_feats is None):
            print("Run reduce_dim() first!!")
            return 
            
        plt.figure(figsize=figsize)
        scatter = plt.scatter(
            self.reduced_feats[:, 0], 
            self.reduced_feats[:, 1], 
            c=self.cluster_labels, 
            cmap='viridis', 
            alpha=0.8,
            s=50
        )
        plt.colorbar(scatter, label='Cluster')
        plt.title(f'Disk Clusters using {self.clustering_method.capitalize()} (n={self.n_clusters})')
        plt.xlabel('Reduced Dimension 1')
        plt.ylabel('Reduced Dimension 2')
        plt.grid(True, alpha=0.3)
        plt.show()

    def plot_cluster(self, samples_per_cluster=10):
        unique_clusters = np.unique(self.cluster_labels)
        for cluster in unique_clusters:
            if (cluster==-1):
                cluster_name = "Noise"
            else:
                cluster_name = f'Cluster {cluster}'
            cluster_indices = np.where(self.cluster_labels == cluster)[0]

            if len(cluster_indices) > 0:
                sample_indices = np.random.choice(
                    cluster_indices, 
                    size=min(samples_per_cluster, len(cluster_indices)), 
                    replace=False
                )
                n = len(sample_indices)
                cols = min(n, 3)
                rows = (n + cols - 1) // cols
                fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
                fig.suptitle(f"{cluster_name} - {len(cluster_indices)} samples", fontsize=16, fontweight = 'bold')
                axes = np.array(axes).reshape(-1)
                for i, idx in enumerate(sample_indices):
                    img = self.images[idx].squeeze()
                    file_name = self.image_paths[idx].split('/')[-1]
                    axes[i].imshow(img)  
                    axes[i].axis("off")  
                    axes[i].set_title(f"File: {file_name}")
        
                for j in range(i + 1, len(axes)):
                    axes[j].axis("off")

                plt.tight_layout()
                plt.show()
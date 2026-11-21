"""
Goal 1: Treat songs as points and find similarities between them.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 1. Load and prepare features
df = pd.read_csv("songs_features.csv")
print(f"Loaded {len(df)} songs, {df.shape[1]} columns")

feature_cols = [c for c in df.columns if c not in ["song_id", "key"]]
X_raw = df[feature_cols].fillna(0).values
X = StandardScaler().fit_transform(X_raw)

# 2. Pairwise similarity
sim_matrix = cosine_similarity(X)
sim_df = pd.DataFrame(sim_matrix, index=df.song_id, columns=df.song_id)
sim_df.to_csv("similarity_matrix.csv")
print("Saved similarity_matrix.csv")

np.fill_diagonal(sim_matrix, -1)
i, j = np.unravel_index(np.argmax(sim_matrix), sim_matrix.shape)
print(f"\nMost similar pair overall: song {df.song_id.iloc[i]} <-> "
      f"song {df.song_id.iloc[j]}  (similarity = {sim_matrix[i, j]:.3f})")

example_idx = 0
example_id = df.song_id.iloc[example_idx]
sims_to_example = sim_matrix[example_idx]
top5_idx = np.argsort(sims_to_example)[::-1][:5]
print(f"\nTop 5 songs most similar to song {example_id}:")
for idx in top5_idx:
    print(f"  song {df.song_id.iloc[idx]}  (similarity = {sims_to_example[idx]:.3f})")

# 3. 2D visualization with PCA
pca = PCA(n_components=2)
coords = pca.fit_transform(X)
print(f"\nPCA explained variance (2 components): {pca.explained_variance_ratio_.sum():.2%}")

# 4. K-means clustering
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X)
df["cluster"] = clusters

print(f"\nCluster sizes (k={n_clusters}):")
print(df["cluster"].value_counts().sort_index())

# 5. Plot
plt.figure(figsize=(8, 6))
scatter = plt.scatter(coords[:, 0], coords[:, 1], c=clusters, cmap="tab10", s=15, alpha=0.7)
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
plt.title("POP909 songs, clustered by chord + melody features")
plt.colorbar(scatter, label="cluster")
plt.tight_layout()
plt.savefig("song_clusters.png", dpi=150)
print("\nSaved song_clusters.png")

df.to_csv("songs_features_with_clusters.csv", index=False)
print("Saved songs_features_with_clusters.csv")

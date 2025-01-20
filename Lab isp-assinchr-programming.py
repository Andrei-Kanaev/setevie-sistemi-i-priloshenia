#!/usr/bin/env python
# coding: utf-8

# # Лабораторная работа 7

# In[7]:


import os
os.environ['OMP_NUM_THREADS'] = '1'

import warnings
warnings.filterwarnings("ignore", message="KMeans is known to have a memory leak")

from sklearn.datasets import load_iris, make_circles, make_blobs
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import numpy as np

def load_data():
    iris = load_iris()
    X_iris = iris.data[:, 2:4]
    return X_iris

def visualize_kmeans(X, n_clusters_list):
    plt.figure(figsize=(12, 4))
    for i, n_clusters in enumerate(n_clusters_list):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(X)
        plt.subplot(1, len(n_clusters_list), i + 1)
        plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis', s=50)
        plt.title(f'K-means with {n_clusters} clusters')
    plt.show()

def visualize_dbscan(X, eps_list, min_samples):
    plt.figure(figsize=(12, 4))
    for i, eps in enumerate(eps_list):
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X)
        plt.subplot(1, len(eps_list), i + 1)
        plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50)
        plt.title(f'DBSCAN (eps={eps}, min_samples={min_samples})')
    plt.show()

def visualize_hierarchical(X):
    linkage_matrix = linkage(X, method='ward')
    plt.figure(figsize=(10, 6))
    dendrogram(linkage_matrix)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Sample index')
    plt.ylabel('Distance')
    plt.show()

def plot_hierarchical_clusters(X):
    hierarchical = AgglomerativeClustering(n_clusters=3)
    hierarchical_labels = hierarchical.fit_predict(X)
    plt.scatter(X[:, 0], X[:, 1], c=hierarchical_labels, cmap='viridis', s=50)
    plt.title('Hierarchical Clustering')
    plt.show()

def cluster_and_evaluate(X):
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans_labels = kmeans.fit_predict(X)

    hierarchical = AgglomerativeClustering(n_clusters=3)
    hierarchical_labels = hierarchical.fit_predict(X)

    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan_labels = dbscan.fit_predict(X)

    kmeans_score = silhouette_score(X, kmeans_labels)
    hierarchical_score = silhouette_score(X, hierarchical_labels)

    dbscan_filtered = X[dbscan_labels != -1]
    dbscan_labels_filtered = dbscan_labels[dbscan_labels != -1]
    dbscan_score = silhouette_score(dbscan_filtered, dbscan_labels_filtered)

    print("Silhouette Scores:")
    print(f"K-means: {kmeans_score:.2f}")
    print(f"Hierarchical: {hierarchical_score:.2f}")
    print(f"DBSCAN: {dbscan_score:.2f}")

def main():
    X_iris = load_data()
    visualize_kmeans(X_iris, [2, 3, 4])
    X_circles, _ = make_circles(n_samples=500, factor=0.5, noise=0.05, random_state=42)
    visualize_dbscan(X_circles, [0.1, 0.2, 0.3], min_samples=5)
    X_blobs, _ = make_blobs(n_samples=100, centers=3, cluster_std=1.0, random_state=42)
    visualize_hierarchical(X_blobs)
    plot_hierarchical_clusters(X_blobs)
    cluster_and_evaluate(X_iris)

if __name__ == "__main__":
    main()


# In[ ]:





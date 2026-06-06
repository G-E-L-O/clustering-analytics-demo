import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import dendrogram, linkage

st.set_page_config(page_title="Análisis de Clustering", layout="centered")

st.title("Análisis de Clustering: K-Means, DBSCAN y Jerárquico")

st.sidebar.header("Parámetros del Entorno")
dataset_type = st.sidebar.selectbox("Conjunto de Datos:", [
    "Segmentación de Clientes (Blobs)", 
    "Rutas Geográficas (Moons)"
])
n_samples = st.sidebar.slider("Volumen de datos (N)", 100, 800, 300)

if "Clientes" in dataset_type:
    X, _ = make_blobs(n_samples=n_samples, centers=4, cluster_std=0.70, random_state=42)
    X[:, 0] = np.interp(X[:, 0], (X[:, 0].min(), X[:, 0].max()), (15, 130))
    X[:, 1] = np.interp(X[:, 1], (X[:, 1].min(), X[:, 1].max()), (5, 95))
    xlabel, ylabel = "Ingreso (Miles $)", "Puntuación de Gasto (1-100)"
else:
    X, _ = make_moons(n_samples=n_samples, noise=0.08, random_state=42)
    X[:, 0] = np.interp(X[:, 0], (X[:, 0].min(), X[:, 0].max()), (-77.10, -76.90))
    X[:, 1] = np.interp(X[:, 1], (X[:, 1].min(), X[:, 1].max()), (-12.10, -11.90))
    xlabel, ylabel = "Longitud", "Latitud"

st.sidebar.header("Configuración del Modelo")
algorithm = st.sidebar.selectbox("Algoritmo:", [
    "K-Means (Iterativo)", 
    "K-Means (Final)", 
    "K-Means (Método del Codo)",
    "DBSCAN",
    "Clustering Jerárquico (Dendrograma)"
])

fig, ax = plt.subplots(figsize=(10, 6.5))
sil_score = -1 

if algorithm == "K-Means (Iterativo)":
    k = st.sidebar.slider("Parámetro K", 2, 6, 4)
    
    np.random.seed(42)
    initial_indices = np.random.choice(X.shape[0], k, replace=False)
    centroids = X[initial_indices]
    
    history_centroids = [centroids.copy()]
    history_labels = []
    
    for i in range(15):
        distances = cdist(X, centroids, 'euclidean')
        labels = np.argmin(distances, axis=1)
        history_labels.append(labels.copy())
        
        new_centroids = np.array([X[labels == j].mean(axis=0) if len(X[labels == j]) > 0 else centroids[j] for j in range(k)])
        history_centroids.append(new_centroids.copy())
        
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
        
    step = st.sidebar.slider("Iteración actual", 0, len(history_labels) - 1, 0)
    current_labels = history_labels[step]
    current_centroids = history_centroids[step]
    
    ax.scatter(X[:, 0], X[:, 1], c=current_labels, cmap='viridis', s=60, edgecolor='k', alpha=0.6)
    ax.scatter(current_centroids[:, 0], current_centroids[:, 1], c='red', s=300, marker='X', edgecolors='white', linewidth=2, label='Centroides')
    ax.set_title(f"K-Means Iteración {step + 1}/{len(history_labels)}", fontsize=14)
    ax.legend()
    
    if len(set(current_labels)) > 1:
        sil_score = silhouette_score(X, current_labels)

elif algorithm == "K-Means (Final)":
    k = st.sidebar.slider("Parámetro K", 2, 8, 4)
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(X)
    
    ax.scatter(X[:, 0], X[:, 1], c=model.labels_, cmap='viridis', s=60, edgecolor='k')
    ax.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], c='red', s=300, marker='X', edgecolors='white', linewidth=2, label='Centroides')
    ax.set_title(f"K-Means Final (K={k})", fontsize=14)
    ax.legend()
    
    if len(set(model.labels_)) > 1:
        sil_score = silhouette_score(X, model.labels_)

elif algorithm == "K-Means (Método del Codo)":
    st.info("💡 Calculando la inercia (WCSS) para múltiples valores de K para encontrar el número óptimo de clusters.")
    inercia = []
    K_range = range(1, 11)
    for k_val in K_range:
        model = KMeans(n_clusters=k_val, random_state=42)
        model.fit(X)
        inercia.append(model.inertia_)
        
    ax.plot(K_range, inercia, marker='o', linestyle='-', color='b', linewidth=2, markersize=8)
    ax.set_title("Método del Codo (Elbow Method) para K-Means", fontsize=14)
    xlabel = "Número de Clusters (K)"
    ylabel = "Inercia (Suma de Errores Cuadráticos)"
    
    # Marcador visual en el codo (K=4 para el Caso A)
    if "Clientes" in dataset_type:
        ax.axvline(x=4, color='red', linestyle='--', label="Codo Óptimo Sugerido (K=4)")
        ax.legend()

elif algorithm == "DBSCAN":
    if "Clientes" in dataset_type:
        eps_val = st.sidebar.slider("Epsilon", 5.0, 30.0, 12.0, step=1.0)
    else:
        eps_val = st.sidebar.slider("Epsilon", 0.01, 0.08, 0.03, step=0.005)
        
    min_samples = st.sidebar.slider("minPts", 2, 15, 5)
    model = DBSCAN(eps=eps_val, min_samples=min_samples)
    model.fit(X)
    labels = model.labels_
    
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    
    core_mask = np.zeros_like(labels, dtype=bool)
    core_mask[model.core_sample_indices_] = True

    for k_label, col in zip(unique_labels, colors):
        xy = X[labels == k_label]
        if k_label == -1:
            ax.plot(xy[:, 0], xy[:, 1], 'x', markerfacecolor='k', markeredgecolor='k', markersize=8, label="Ruido")
        else:
            core_pts = xy[core_mask[labels == k_label]]
            border_pts = xy[~core_mask[labels == k_label]]
            ax.plot(core_pts[:, 0], core_pts[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=9, label=f"Cluster {k_label}")
            ax.plot(border_pts[:, 0], border_pts[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='none', markersize=4, alpha=0.6)

    ax.set_title(f"DBSCAN (eps={eps_val}, minPts={min_samples})", fontsize=14)
    
    if len(unique_labels) > 1 and not (len(unique_labels) == 2 and -1 in unique_labels):
        sil_score = silhouette_score(X, labels)

elif algorithm == "Clustering Jerárquico (Dendrograma)":
    method = st.sidebar.selectbox("Método de Enlace:", ["ward", "complete", "average", "single"])
    
    if n_samples > 100:
        st.caption("💡 Se muestra una submuestra de 100 datos para que las hojas del árbol sean legibles.")
        np.random.seed(42)
        X_sample = X[np.random.choice(X.shape[0], 100, replace=False)]
    else:
        X_sample = X
        
    Z = linkage(X_sample, method=method)
    dendrogram(Z, ax=ax)
    
    ax.set_title(f"Dendrograma Jerárquico (Enlace: {method})", fontsize=14)
    xlabel = "Índice de la Observación"
    ylabel = "Distancia de Fusión"
    sil_score = -1

ax.set_xlabel(xlabel)
ax.set_ylabel(ylabel)
ax.grid(True, linestyle='--', alpha=0.4)
st.pyplot(fig)

st.markdown("---")
if sil_score != -1:
    st.metric(label="Coeficiente de Silhouette", value=f"{sil_score:.3f}")
elif algorithm not in ["Clustering Jerárquico (Dendrograma)", "K-Means (Método del Codo)"]:
    st.metric(label="Coeficiente de Silhouette", value="N/A")

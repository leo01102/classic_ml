# kmeans.py

import csv
import sys
import os
import random
import math
import matplotlib.pyplot as plt

def load_data(path, header=True):
    X = []
    with open(path) as f:
        reader = csv.reader(f)
        if header:
            next(reader)
        for row in reader:
            if not row:
                continue
            try:
                vals = [float(v) for v in row]
                X.append(vals)
            except ValueError:
                print(f"Fila inválida (omitida): {row}")
    return X

def euclidean(a, b):
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))

def initialize_centroids(X, k):
    return random.sample(X, k)

def assign_clusters(X, centroids):
    clusters = [[] for _ in centroids]
    for x in X:
        distances = [euclidean(x, c) for c in centroids]
        idx = distances.index(min(distances))
        clusters[idx].append(x)
    return clusters

def recompute_centroids(clusters):
    new_centroids = []
    for cluster in clusters:
        if not cluster:
            new_centroids.append(None)
            continue
        m = len(cluster[0])
        mean = [sum(point[i] for point in cluster) / len(cluster) for i in range(m)]
        new_centroids.append(mean)
    return new_centroids

def kmeans(X, k, max_iter=100, tol=1e-4):
    centroids = initialize_centroids(X, k)
    for i in range(max_iter):
        clusters = assign_clusters(X, centroids)
        new_centroids = recompute_centroids(clusters)
        # reemplazar vacíos
        for idx, c in enumerate(new_centroids):
            if c is None:
                new_centroids[idx] = random.choice(X)
        shifts = [euclidean(c, nc) for c, nc in zip(centroids, new_centroids)]
        centroids = new_centroids
        if max(shifts) < tol:
            print(f"Converged en iteración {i}")
            break
    return centroids, clusters

def plot_clusters_2d(X, clusters, centroids):
    colors = plt.get_cmap('tab10')
    plt.figure()
    for idx, cluster in enumerate(clusters):
        xs = [p[0] for p in cluster]
        ys = [p[1] for p in cluster]
        plt.scatter(xs, ys, s=30, color=colors(idx), label=f'Cluster {idx}')
    for idx, c in enumerate(centroids):
        plt.scatter(c[0], c[1], marker='X', s=200, color=colors(idx), edgecolor='black', linewidth=2)
    plt.title('K-means Clustering (2D)')
    plt.xlabel('Dim 1')
    plt.ylabel('Dim 2')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    print("--- K-means Interactivo ---")
    path = input("Ruta al archivo CSV: ").strip()
    if not os.path.exists(path):
        print("No se encontró el archivo.")
        sys.exit(1)
    X = load_data(path)
    if not X:
        print("No hay datos válidos en el archivo.")
        sys.exit(1)
    dim = len(X[0])

    # Parámetros
    k = int(input("Número de clusters (k): ").strip())
    max_iter_input = input("Máximo de iteraciones (Enter=100): ").strip()
    max_iter = int(max_iter_input) if max_iter_input else 100
    tol_input = input("Tolerancia de convergencia (Enter=1e-4): ").strip()
    tol = float(tol_input) if tol_input else 1e-4

    print("===== RESULTADOS =====")
    centroids, clusters = kmeans(X, k, max_iter, tol)
    print("Centroides finales:")
    for idx, c in enumerate(centroids):
        print(f"  Cluster {idx}: {c}")
    sizes = [len(cl) for cl in clusters]
    print(f"Tamaño de cada cluster: {sizes}")

    if dim == 2:
        plot_clusters_2d(X, clusters, centroids)
    else:
        print(f"Atención: no se grafica porque los datos no son 2D (dimensión = {dim}).")

    print("Proceso completado.")

if __name__ == '__main__':
    main()

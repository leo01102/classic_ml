# kmeans.py

import csv
import sys
import os
import random
import math
import matplotlib.pyplot as plt
from rich import print
from rich.console import Console
from rich.prompt import Prompt, IntPrompt, FloatPrompt
from rich.panel import Panel

console = Console()


def load_data(path, header=True):
    with open(path) as f:
        reader = csv.reader(f)
        headers = next(reader) if header else None
        numeric_indices = None
        text_attr = None
        X = []
        for row in reader:
            if not row:
                continue
            if numeric_indices is None:
                numeric_indices = []
                for i, v in enumerate(row):
                    try:
                        float(v)
                        numeric_indices.append(i)
                    except ValueError:
                        if text_attr is None and headers:
                            text_attr = headers[i].lower()
                if not numeric_indices:
                    console.print(f"[bold red]Error:[/] No se detectaron columnas numéricas en {row}")
                    sys.exit(1)
                num_col_names = [headers[i].lower() for i in numeric_indices] if headers else [f"feature_{j}" for j in numeric_indices]
            try:
                numeric_vals = [float(row[i]) for i in numeric_indices]
                X.append(numeric_vals)
            except Exception:
                console.print(f"[yellow]Fila inválida (omitida):[/] {row}")
        return X, num_col_names, text_attr or "punto"


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
        for idx, c in enumerate(new_centroids):
            if c is None:
                new_centroids[idx] = random.choice(X)
        shifts = [euclidean(c, nc) for c, nc in zip(centroids, new_centroids)]
        centroids = new_centroids
        if max(shifts) < tol:
            console.print(f"[green]🎉 Convergió en iteración {i}[/green]")
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
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    # Título
    console.rule("[bold cyan]📊 K-means Interactivo[/bold cyan]")
    raw = Prompt.ask("📂 Ruta al archivo CSV (o nombre dentro de 'data')").strip()
    if os.path.exists(raw):
        path = raw
    else:
        default = os.path.join('data', raw)
        if os.path.exists(default):
            path = default
        else:
            console.print(f"[red]Archivo no encontrado ni en '{raw}' ni en '{default}'[/red]")
            sys.exit(1)

    X, col_names, text_attr = load_data(path)
    if not X:
        console.print("[bold red]No hay datos válidos en el archivo.[/bold red]")
        sys.exit(1)

    dim = len(col_names)
    k = IntPrompt.ask("🔢 Número de clusters (k)")
    max_iter = IntPrompt.ask("🔁 Máximo iteraciones", default=100)
    tol = FloatPrompt.ask("🎯 Tolerancia convergencia", default=1e-4)

    console.rule("[bold green]📈 Resultados[/bold green]")
    centroids, clusters = kmeans(X, k, max_iter, tol)

    console.print("[bold magenta]Centroides finales:[/bold magenta]")
    for idx, c in enumerate(centroids):
        console.print(f"[cyan]  Cluster {idx}[/cyan]: {c}")

    sizes = [len(cl) for cl in clusters]
    console.print(f"[bold blue]Tamaño de cada cluster:[/bold blue] {sizes}")

    if dim == 2:
        plot_clusters_2d(X, clusters, centroids)
    else:
        console.print(f"[yellow]⚠️ Atención:[/] no se grafica, dimensión = {dim} ≠ 2.")

    console.print("[green]✅ Proceso completado.[/green]")

    # Clasificación de nuevos puntos
    while True:
        ans = Prompt.ask(f"¿Deseas clasificar nuevo {text_attr}?", choices=["s","n"], default="n")
        if ans != 's':
            break
        vals = Prompt.ask(f"✏️ Ingresa [bold]{', '.join(col_names)}[/bold] separados por coma").split(',')
        try:
            punto = [float(v) for v in vals]
            if len(punto) != dim:
                raise ValueError
            distancias = [euclidean(punto, c) for c in centroids]
            idx_c = distancias.index(min(distancias))
            console.print(Panel.fit(
                f"➡️ El {text_attr} pertenece al [bold green]cluster {idx_c}[/bold green]\n"
                f"[dim]Centroide: {centroids[idx_c]}[/dim]",
                title="Clasificación", border_style="green"
            ))
        except:
            console.print("[red]Entrada inválida.[/red]")

if __name__ == '__main__':
    main()

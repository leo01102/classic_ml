# linear_regression.py

import csv
import sys
import os
import matplotlib.pyplot as plt
from rich.console import Console
from rich.prompt import Prompt, FloatPrompt
from rich.panel import Panel
from rich.markdown import Markdown

console = Console()


def load_data(path, header=True):
    """
    Carga datos de regresión lineal desde CSV.
    Asume última columna como target, resto features.
    Agrega bias=1.0 en X.
    """
    X, y = [], []
    with open(path) as f:
        reader = csv.reader(f)
        if header:
            next(reader)
        for row in reader:
            if not row:
                continue
            *features, target = row
            try:
                x_vals = [float(v) for v in features]
                y_val = float(target)
                X.append([1.0] + x_vals)
                y.append(y_val)
            except ValueError:
                console.print(f"[yellow]Fila inválida (omitida):[/] {row}")
    return X, y


def transpose(M):
    return list(map(list, zip(*M)))


def matmul(A, B):
    return [[sum(a * b for a, b in zip(row, col)) for col in zip(*B)] for row in A]


def add_regularization(XTX, lam):
    n = len(XTX)
    reg = [[lam if i == j and i != 0 else 0 for j in range(n)] for i in range(n)]
    return [[XTX[i][j] + reg[i][j] for j in range(n)] for i in range(n)]


def invert_matrix(M):
    n = len(M)
    aug = [row[:] + [float(i == j) for j in range(n)] for i, row in enumerate(M)]
    for i in range(n):
        if abs(aug[i][i]) < 1e-12:
            for r in range(i + 1, n):
                if abs(aug[r][i]) > abs(aug[i][i]):
                    aug[i], aug[r] = aug[r], aug[i]
                    break
        pivot = aug[i][i]
        if abs(pivot) < 1e-12:
            raise ValueError("Matriz singular")
        aug[i] = [v / pivot for v in aug[i]]
        for r in range(n):
            if r != i:
                factor = aug[r][i]
                aug[r] = [vr - factor * vi for vr, vi in zip(aug[r], aug[i])]
    return [row[n:] for row in aug]


def fit_linear_regression(X, y, lam=0.0):
    X_T = transpose(X)
    XTX = matmul(X_T, X)
    if lam > 0.0:
        XTX = add_regularization(XTX, lam)
    XTy = matmul(X_T, [[v] for v in y])
    XTX_inv = invert_matrix(XTX)
    w = matmul(XTX_inv, XTy)
    return [wi[0] for wi in w]


def predict(w, x):
    return sum(wi * xi for wi, xi in zip(w, x))


def plot_regression(X, y, w, x_new=None, y_new=None):
    xs = [row[1] for row in X]
    ys = y
    plt.figure()
    plt.scatter(xs, ys, marker='o', label='Datos')
    x_line = [min(xs), max(xs)]
    y_line = [predict(w, [1, x]) for x in x_line]
    plt.plot(x_line, y_line, 'r--', label='Regresión')
    if x_new is not None and y_new is not None:
        plt.scatter([x_new], [y_new], marker='s', color='green', s=100, label='Predicción')
    plt.title('Regresión Lineal')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    console.rule("[bold cyan]Regresión lineal[/bold cyan]")
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

    try:
        X, y = load_data(path)
    except Exception as e:
        console.print(f"[red]Error cargando datos:[/] {e}")
        sys.exit(1)

    n_feat = len(X[0]) - 1
    lam = FloatPrompt.ask("🔒 Lambda de regularización (0=ninguna)", default=0.0)

    w = fit_linear_regression(X, y, lam)
    console.print(Panel.fit(
        f"💡 Coeficientes (bias primero): [bold]{[f'{wi:.4f}' for wi in w]}[/]",
        title="Resultados", border_style="green"
    ))

    if Prompt.ask("¿Deseas predecir un nuevo punto?", choices=["s","n"], default="n") == "s":
        entrada = Prompt.ask(f"✏️ Ingresa {n_feat} valores separados por coma").strip()
        try:
            vals = [float(v) for v in entrada.split(',')]
            if len(vals) != n_feat:
                raise ValueError
            x_vec = [1.0] + vals
            y_pred = predict(w, x_vec)
            console.print(Panel.fit(
                f"➡️ Predicción para {vals}: [bold]{y_pred:.4f}[/]",
                title="Predicción", border_style="cyan"
            ))
        except:
            console.print("[red]Entrada inválida.[/red]")
    else:
        x_vec, y_pred = None, None

    if n_feat == 1:
        plot_regression(X, y, w, x_vec[1] if x_vec else None, y_pred)
    else:
        console.print(f"[yellow]Atención:[/] no se grafica porque hay {n_feat} características.")

if __name__ == '__main__':
    main()

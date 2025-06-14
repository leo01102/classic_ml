# perceptron.py

import csv
import sys
import os
import random
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt, IntPrompt, FloatPrompt
from rich.panel import Panel
from rich.markdown import Markdown
import matplotlib.pyplot as plt

console = Console()


def load_data(path, header=True):
    X, y = [], []
    with open(path) as f:
        reader = csv.reader(f)
        if header:
            next(reader)
        for row in reader:
            if not row:
                continue
            *features, label = row
            try:
                x = [float(v) for v in features]
            except ValueError:
                console.print(f"[red]Error:[/] característica no numérica en fila: {row}")
                sys.exit(1)
            X.append(x)
            y.append(label)
    return X, y


def train_test_split(X, y, test_ratio=0.2, shuffle=True, seed=42):
    n = len(X)
    indices = list(range(n))
    if shuffle:
        random.seed(seed)
        random.shuffle(indices)
    split = int(n * (1 - test_ratio))
    train_idx, test_idx = indices[:split], indices[split:]
    return ([X[i] for i in train_idx], [X[i] for i in test_idx],
            [y[i] for i in train_idx], [y[i] for i in test_idx])


def initialize_weights(n_features, init_str=None, random_range=None):
    if init_str:
        parts = init_str.split(',')
        if len(parts) != n_features + 1:
            console.print(f"[red]Error:[/] se esperaban {n_features+1} valores, recibidos {len(parts)}")
            sys.exit(1)
        w = [float(v) for v in parts]
        return w[1:], w[0]
    if random_range:
        low, high = random_range
        return ([random.uniform(low, high) for _ in range(n_features)],
                random.uniform(low, high))
    return [0.0] * n_features, 0.0


def predict(weights, bias, x):
    activation = sum(w * xi for w, xi in zip(weights, x)) + bias
    return activation, (1 if activation >= 0 else -1)


def predict_binary(weights, bias, x, threshold=0.0):
    activation = sum(w * xi for w, xi in zip(weights, x)) + bias
    return activation, (1 if activation >= threshold else 0)


def train_perceptron(X, y_bin, lr=0.1, epochs=100, init_w=None, init_b=None):
    weights = init_w[:] if init_w else [0.0] * len(X[0])
    bias = init_b if init_b is not None else 0.0
    for epoch in range(epochs):
        errors = 0
        for xi, yi in zip(X, y_bin):
            _, pred = predict(weights, bias, xi)
            if pred != yi:
                update = lr * yi
                weights = [w + update * xi[i] for i, w in enumerate(weights)]
                bias += update
                errors += 1
        if errors == 0:
            return weights, bias, epoch
    return weights, bias, None


def evaluate_accuracy(weights, bias, X, y_bin):
    hits = sum(predict(weights, bias, xi)[1] == yi for xi, yi in zip(X, y_bin))
    return 100.0 * hits / len(X) if X else 0.0


def test_weight_set(weights, bias, X, y_true, threshold=0.0):
    results, ok = [], True
    for xi, yi in zip(X, y_true):
        act, pred = predict_binary(weights, bias, xi, threshold)
        if pred != yi:
            ok = False
        results.append((xi, act, pred, yi))
    return ok, results


def grid_search(X, y_true, bias, w0_rng, w1_rng, threshold=0.0):
    sol = []
    for w0 in frange(*w0_rng):
        for w1 in frange(*w1_rng):
            ws = [w1] if len(X[0]) == 1 else [w0, w1]
            ok, _ = test_weight_set(ws, bias, X, y_true, threshold)
            if ok:
                sol.append((w0, w1))
    return sol


def frange(start, stop, step):
    x = start
    while x <= stop:
        yield x
        x += step


def plot_decision_boundary(X, y, weights, bias):
    x_vals = [p[0] for p in X]
    y_vals = [p[1] for p in X]
    pos = [(p[0], p[1]) for p, lab in zip(X, y) if lab == 1]
    neg = [(p[0], p[1]) for p, lab in zip(X, y) if lab == -1]
    plt.figure()
    if pos:
        plt.scatter(*zip(*pos), marker='o', label='+1')
    if neg:
        plt.scatter(*zip(*neg), marker='x', label='-1')
    w1, w2 = weights
    if w2 != 0:
        xmin, xmax = min(x_vals) - 1, max(x_vals) + 1
        plt.plot([xmin, xmax], [-(w1 * xmin + bias) / w2, -(w1 * xmax + bias) / w2], 'r--')
    else:
        plt.axvline(-bias / w1, color='r', linestyle='--')
    plt.legend(); plt.grid(); plt.show()

# ---- RUN MODES ----

def run_binary(X_all, y_all, n_feat):
    console.rule("[bold cyan]Entrenamiento Binario[/bold cyan]")
    table = Table()
    table.add_column("Clases disponibles", style="cyan")
    for cls in sorted(set(y_all)):
        table.add_row(cls)
    console.print(table)
    pos = Prompt.ask("Clase positiva")
    y_bin = [1 if y == pos else -1 for y in y_all]
    Xtr, Xte, ytr, yte = train_test_split(X_all, y_bin)
    lr = FloatPrompt.ask("Tasa de aprendizaje", default=0.1)
    epochs = IntPrompt.ask("Épocas", default=100)
    wi, bi = prompt_weights(n_feat)
    wf, bf, conv = train_perceptron(Xtr, ytr, lr, epochs, wi, bi)
    console.rule("[bold green]Resultados[/bold green]")
    print_results_binary(wf, bf, Xtr, ytr, Xte, yte, conv)
    classify_new_point(wf, bf, n_feat, pos)


def run_ovr(X_all, y_all, n_feat):
    console.rule("[bold magenta]Entrenamiento One-vs-Rest[/bold magenta]")
    table = Table()
    table.add_column("Clases disponibles", style="magenta")
    for cls in sorted(set(y_all)):
        table.add_row(cls)
    console.print(table)
    Xtr, Xte, ytr_all, yte_all = train_test_split(X_all, y_all)
    lr = FloatPrompt.ask("Tasa de aprendizaje", default=0.1)
    epochs = IntPrompt.ask("Épocas", default=100)
    classifiers = {}
    for cls in sorted(set(y_all)):
        wi, bi = prompt_weights(n_feat)
        ytr = [1 if y == cls else -1 for y in ytr_all]
        yte = [1 if y == cls else -1 for y in yte_all]
        w, b, conv = train_perceptron(Xtr, ytr, lr, epochs, wi, bi)
        console.print(f"[green][{cls}] converged={conv}[/] pesos={w}, bias={b:.4f}, "
                      f"acc_train={evaluate_accuracy(w,b,Xtr,ytr):.2f}%")
        classifiers[cls] = (w, b)
    pred = classify_new_multiclass(classifiers, n_feat)
    console.print(Panel(f":telescope: Predicción multiclase: [bold yellow]{pred}[/bold yellow]", title="Clasificación", border_style="yellow"))


def run_test(X_all, y_all, n_feat):
    console.rule("[bold blue]Probar Pesos Específicos[/bold blue]")
    table = Table()
    table.add_column("Clases disponibles", style="blue")
    for cls in sorted(set(y_all)):
        table.add_row(cls)
    console.print(table)
    pos = Prompt.ask("Clase positiva")
    bias = FloatPrompt.ask("Bias", default=1.0)
    th = FloatPrompt.ask("Umbral decisión", default=0.5)
    ws = [float(v) for v in Prompt.ask(f"Ingresa {n_feat} pesos separados por coma").split(',')]
    ok, res = test_weight_set(ws, bias, X_all, [1 if y==pos else 0 for y in y_all], th)
    if ok:
        console.print(Panel("[white_check_mark] [green]Los pesos clasifican correctamente todos los ejemplos.[/green]", title="Éxito", border_style="green"))
    else:
        console.print(Panel("[x] [red]Los pesos NO clasifican correctamente.[/red]", title="Fallo", border_style="red"))
        for xi, act, pred, yi in res:
            console.print(f"x={xi} -> activación={act:.3f}, pred={pred}, actual={yi}")


def run_grid(X_all, y_all, n_feat):
    console.rule("[bold yellow]Búsqueda en Grilla[/bold yellow]")
    table = Table()
    table.add_column("Clases disponibles", style="yellow")
    for cls in sorted(set(y_all)):
        table.add_row(cls)
    console.print(table)
    pos = Prompt.ask("Clase positiva")
    bias = FloatPrompt.ask("Bias", default=1.0)
    th = FloatPrompt.ask("Umbral decisión", default=0.5)
    w0r = tuple(map(float, Prompt.ask("Rango w0 start,stop,step").split(',')))
    w1r = tuple(map(float, Prompt.ask("Rango w1 start,stop,step").split(',')))
    sols = grid_search(X_all, [1 if y==pos else 0 for y in y_all], bias, w0r, w1r, th)
    if sols:
        console.print(Panel("\n".join(f"({w0:.3f}, {w1:.3f})" for w0, w1 in sols), title="Soluciones", border_style="green"))
    else:
        console.print(Panel("No se encontraron soluciones.", title="Soluciones", border_style="red"))


def prompt_weights(n_feat):
    # Selección del método de inicialización
    console.print("[bold]Inicialización de pesos disponibles:[/bold] 1) Ceros  2) Aleatorio en rango")
    choice = Prompt.ask("Seleccione el método de inicialización de pesos", choices=["1","2"], default="1")
    if choice == "2":
        r = Prompt.ask("Rango (low,high)")
        try:
            low, high = map(float, r.split(','))
        except:
            low, high = -0.01, 0.01
        return initialize_weights(n_feat, None, (low, high))
    return initialize_weights(n_feat)


def print_results_binary(w, b, Xtr, ytr, Xte, yte, conv):
    console.print(Panel.fit(
        f"Converged en epoch [bold]{conv}[/bold]\n"
        f"Pesos: {w}\n"
        f"Bias: {b:.4f}\n"
        f"Acc train: {evaluate_accuracy(w,b,Xtr,ytr):.2f}%\n"
        f"Acc test: {evaluate_accuracy(w,b,Xte,yte):.2f}%",
        title="Resultados", border_style="green"
    ))
    if len(w) == 2:
        plot_decision_boundary(Xtr, ytr, w, b)


def classify_new_point(w, b, n_feat, pos_label):
    if Prompt.ask("¿Clasificar nuevo punto?", choices=["s","n"], default="n") == "s":
        vals = [float(v) for v in Prompt.ask(f"Ingresa {n_feat} valores separados por coma").split(',')]
        act, pred = predict(w, b, vals)
        lbl = pos_label if pred == 1 else f"not {pos_label}"
        console.print(Panel.fit(
            f"➡️ Clasificación: [bold]{lbl}[/bold] (activación={act:.3f})",
            title="Clasificación", border_style="cyan"
        ))


def classify_new_multiclass(classifiers, n_feat):
    if Prompt.ask("¿Clasificar nuevo punto?", choices=["s","n"], default="n") == "s":
        vals = [float(v) for v in Prompt.ask(f"Ingresa {n_feat} valores separados por coma").split(',')]
        acts = {c: predict(w,b,vals)[0] for c,(w,b) in classifiers.items()}
        pred = max(acts, key=acts.get)
        return pred
    return None


def main():
    raw = Prompt.ask("Ruta al archivo CSV (o nombre dentro de 'data')")
    if os.path.exists(raw):
        path = raw
    else:
        default = os.path.join('data', raw)
        if os.path.exists(default):
            path = default
        else:
            console.print(f"[red]Archivo no encontrado ni en '{raw}' ni en '{default}'[/red]")
            sys.exit(1)

    X_all, y_all = load_data(path)
    n_feat = len(X_all[0])

    console.rule("[bold cyan]Perceptrón Interactivo[/bold cyan]")
    console.print(Markdown("**Modos disponibles:**\n1) Entrenamiento binario (binary)\n2) One-vs-Rest multiclase (ovr)\n3) Probar pesos manuales (test)\n4) Búsqueda en grilla para pesos (grid)"))
    mode = Prompt.ask("Seleccione modo", choices=["1","2","3","4","binary","ovr","test","grid"])
    dispatch = {
        '1': run_binary, 'binary': run_binary,
        '2': run_ovr,    'ovr': run_ovr,
        '3': run_test,   'test': run_test,
        '4': run_grid,   'grid': run_grid
    }
    func = dispatch.get(mode)
    if func:
        func(X_all, y_all, n_feat)
    else:
        console.print("[red]Modo inválido.[/red]")

if __name__ == '__main__':
    main()

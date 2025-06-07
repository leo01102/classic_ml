# perceptron.py

import csv
import sys
import os
import random
import matplotlib.pyplot as plt

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
                print(f"Error: característica no numérica en fila: {row}")
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
    split_at = int(n * (1 - test_ratio))
    idx_train = indices[:split_at]
    idx_test = indices[split_at:]
    X_train = [X[i] for i in idx_train]
    y_train = [y[i] for i in idx_train]
    X_test = [X[i] for i in idx_test]
    y_test = [y[i] for i in idx_test]
    return X_train, X_test, y_train, y_test

def initialize_weights(n_features, init_str=None, random_range=None):
    if init_str:
        parts = init_str.split(',')
        if len(parts) != n_features + 1:
            print(f"Error: se esperaban {n_features + 1} valores, recibidos {len(parts)}")
            sys.exit(1)
        try:
            all_w = [float(v) for v in parts]
            return all_w[1:], all_w[0]
        except ValueError:
            print("Error: pesos inválidos")
            sys.exit(1)
    elif random_range:
        low, high = random_range
        weights = [random.uniform(low, high) for _ in range(n_features)]
        bias = random.uniform(low, high)
        return weights, bias
    else:
        return [0.0] * n_features, 0.0

def predict(weights, bias, x):
    activation = sum(w * xi for w, xi in zip(weights, x)) + bias
    return activation, (1 if activation >= 0 else -1) # NOTE: para devolver la clase binaria 0 en vez de –1, usar: return activation, (1 if activation >= 0 else 0)

def train_perceptron(X, y_bin, lr=0.1, epochs=100, init_weights=None, init_bias=None):
    n = len(X[0])
    weights = init_weights[:] if init_weights else [0.0] * n
    bias = init_bias if init_bias is not None else 0.0
    for epoch in range(epochs):
        errors = 0
        for xi, yi in zip(X, y_bin):
            _, y_pred = predict(weights, bias, xi)
            if yi != y_pred:
                update = lr * yi
                weights = [w + update * xi[i] for i, w in enumerate(weights)]
                bias += update
                errors += 1
        if errors == 0:
            return weights, bias, epoch
    return weights, bias, None

def evaluate_accuracy(weights, bias, X, y_bin):
    correct = sum(1 for xi, yi in zip(X, y_bin) if predict(weights, bias, xi)[1] == yi)
    return 100.0 * correct / len(X)

def plot_decision_boundary(X, y_bin, weights, bias, title="Perceptrón", x_new=None, pred_new=None):
    x_vals = [p[0] for p in X]
    y_vals = [p[1] for p in X]
    pos_x = [p[0] for p, label in zip(X, y_bin) if label == 1]
    pos_y = [p[1] for p, label in zip(X, y_bin) if label == 1]
    neg_x = [p[0] for p, label in zip(X, y_bin) if label == -1]
    neg_y = [p[1] for p, label in zip(X, y_bin) if label == -1]
    plt.figure()
    plt.scatter(pos_x, pos_y, marker='o', label='+1')
    plt.scatter(neg_x, neg_y, marker='x', label='-1')
    w1, w2 = weights
    if w2 != 0:
        x_min, x_max = min(x_vals) - 1, max(x_vals) + 1
        y_min = (-(w1 * x_min + bias) / w2)
        y_max = (-(w1 * x_max + bias) / w2)
        plt.plot([x_min, x_max], [y_min, y_max], 'r--', label='Frontera')
    else:
        plt.axvline(x=-bias / w1, color='r', linestyle='--', label='Frontera')
    if x_new and pred_new is not None:
        color = 'green' if pred_new == 1 else 'red'
        plt.scatter([x_new[0]], [x_new[1]], marker='s', s=100, color=color)
    plt.title(title)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    path = input("Ruta al archivo CSV: ").strip()
    X_all, y_all = load_data(path, header=True)
    n_features = len(X_all[0])
    mode = input("Modo ('binary' u 'ovr'): ").strip().lower()

    print("Opciones de inicialización de pesos:")
    print("  1) Todos ceros")
    print("  2) Aleatorios en rango")
    choice = input("Seleccione (1/2): ").strip()
    random_range = None
    init_str = None
    if choice == '1':
        init_str = None
    elif choice == '2':
        r = input("Rango de pesos (bajo,alto), ej: -0.05,0.05 (Enter para [-0.01,0.01]): ").strip()
        if r:
            try:
                low_str, high_str = r.split(',')
                low = float(low_str)
                high = float(high_str)
                random_range = (low, high)
            except:
                print("Rango inválido, usando [-0.01, 0.01] por defecto.")
                random_range = (-0.01, 0.01)
        else:
            random_range = (-0.01, 0.01)
    else:
        print("Opción inválida, se usará inicialización cero.")
        init_str = None

    if mode == 'binary':
        print("Clases disponibles:", sorted(set(y_all)))
        pos_class = input("Clase positiva: ").strip()
        y_bin = [1 if label == pos_class else -1 for label in y_all] # NOTE: para usar las clases binarias 0/1 en lugar de –1/+1, reemplazar por y_bin = [1 if label==pos_class else 0 for label in y_all]
        X_train, X_test, y_train, y_test = train_test_split(X_all, y_bin)
        lr = float(input("Tasa de aprendizaje (ej. 0.1): ") or 0.1)
        epochs = int(input("Épocas (ej. 100): ") or 100)

        if init_str:
            init_weights, init_bias = initialize_weights(n_features, init_str, None)
        else:
            init_weights, init_bias = initialize_weights(n_features, None, random_range)

        weights, bias, epoch_conv = train_perceptron(X_train, y_train, lr, epochs, init_weights, init_bias)
        print("===== RESULTADOS =====")
        if epoch_conv is not None:
            print(f"Converged en epoch {epoch_conv}")
        else:
            print(f"No convergió en {epochs} épocas.")
        print(f"Pesos finales: {[f'{w:.4f}' for w in weights]}")
        print(f"Bias final:   {bias:.4f}")
        print(f"Precisión train: {evaluate_accuracy(weights, bias, X_train, y_train):.2f}%")
        print(f"Precisión test:  {evaluate_accuracy(weights, bias, X_test, y_test):.2f}%")

        if n_features == 2:
            plot_decision_boundary(X_train, y_train, weights, bias)
        else:
            print("Atención: no se graficará porque el espacio de características no es 2D.")

        if input("¿Deseás clasificar un nuevo punto? (s/n): ").lower().startswith('s'):
            punto = input(f"Ingresa {n_features} valores separados por coma: ").strip()
            try:
                x = [float(v) for v in punto.split(',')]
                act, pred = predict(weights, bias, x)
                label = pos_class if pred == 1 else f"not {pos_class}"
                print("===== RESULTADOS =====")
                print(f"Clasificación: {label} (activación={act:.3f})")
            except:
                print("Entrada inválida.")

    elif mode == 'ovr':
        classes = sorted(set(y_all))
        X_train, X_test, y_train_all, y_test_all = train_test_split(X_all, y_all)
        lr = float(input("Tasa de aprendizaje (ej. 0.1): ") or 0.1)
        epochs = int(input("Épocas (ej. 100): ") or 100)

        print("===== RESULTADOS =====")
        classifiers = {}
        for cls in classes:
            y_train_bin = [1 if y == cls else -1 for y in y_train_all]
            y_test_bin = [1 if y == cls else -1 for y in y_test_all]
            if init_str:
                init_weights, init_bias = initialize_weights(n_features, init_str, None)
            else:
                init_weights, init_bias = initialize_weights(n_features, None, random_range)
            ws, b, conv = train_perceptron(X_train, y_train_bin, lr, epochs, init_weights, init_bias)
            if conv is not None:
                print(f"[{cls}] Converged en epoch {conv}")
            else:
                print(f"[{cls}] No convergió en {epochs} épocas.")
            print(f"[{cls}] Pesos={[f'{w:.4f}' for w in ws]}, Bias={b:.4f}")
            acc_train = evaluate_accuracy(ws, b, X_train, y_train_bin)
            acc_test = evaluate_accuracy(ws, b, X_test, y_test_bin)
            print(f"[{cls}] Precisión train: {acc_train:.2f}%  |  Precisión test: {acc_test:.2f}%")
            print("––––––––––––––––––––––––––––––––––––––––––––––––––")
            classifiers[cls] = (ws, b)

        if input("¿Deseás clasificar un nuevo punto? (s/n): ").lower().startswith('s'):
            punto = input(f"Ingresa {n_features} valores separados por coma: ").strip()
            try:
                x = [float(v) for v in punto.split(',')]
                acts = {c: predict(ws_, b_, x)[0] for c, (ws_, b_) in classifiers.items()}
                pred_class = max(acts, key=acts.get)
                print("===== RESULTADOS =====")
                print(f"Predicción multiclase: {pred_class} (activación={acts[pred_class]:.3f})")
            except:
                print("Entrada inválida.")

    else:
        print("Modo inválido. Usá 'binary' u 'ovr'.")

if __name__ == '__main__':
    main()

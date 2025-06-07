# Aprendizaje Automático Clásico

Este repositorio incluye implementaciones simples de tres algoritmos fundamentales de Machine Learning:

- **Perceptrón** (`perceptron.py`)
- **Regresión Lineal** (`linear_regression.py`)
- **K-Means** (`kmeans.py`)

Todos los scripts se ejecutan de forma interactiva en la terminal y pueden trabajar con archivos `.csv` ubicados en la carpeta `/data`.

---

## 📋 Requisitos

- Python 3.6 o superior  
- `numpy`  
- `matplotlib`

Instálalos con:

```bash
pip install numpy matplotlib
```

---

## 🚀 Uso de los scripts

### 1. Perceptrón

```bash
python perceptron.py
```

- Ingresá la ruta al CSV (por ejemplo: `data/iris.csv`).  
- Elegí el modo `binary` (binario) u `ovr` (One-vs-Rest multiclase).  
- Configura tasa de aprendizaje, épocas e inicialización de pesos.  
- Si el dataset es 2D, mostrará la frontera de decisión.

### 2. Regresión Lineal

```bash
python linear_regression.py
```

- Ingresá la ruta al CSV.  
- Elegí lambda para regularización (0 = sin regularización).  
- Verás los coeficientes (bias + pesos) y, si es univariada, se graficará la recta.

### 3. K-Means

```bash
python kmeans.py
```

- Ingresá la ruta al CSV.  
- Especificá el número de clusters `k`.  
- Si los datos son 2D, se mostrará la distribución de clusters y centroides.

---

## 📄 Licencia

Este proyecto está bajo **MIT License**.  

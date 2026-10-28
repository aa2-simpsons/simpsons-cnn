# simpsons-cnn

Clasificador de personajes de Los Simpson con redes neuronales convolucionales en PyTorch.

Práctica 1 de Aprendizaje Automático II (2026-27).

## Dataset

[The Simpsons Characters Data](https://www.kaggle.com/datasets/alexattia/the-simpsons-characters-dataset) (Kaggle). Se descarga manualmente y se descomprime en la carpeta `data/` (no se sube al repositorio):

- `data/simpsons_dataset/`: una carpeta por personaje. Se usa para entrenamiento (85%) y validación (15%).
- `data/kaggle_simpson_testset/kaggle_simpson_testset/`: conjunto de test independiente (496 imágenes de los 10 personajes).

Se utilizan los 10 personajes con más imágenes: Homer, Ned Flanders, Moe, Lisa, Bart, Marge, Krusty, Skinner, Sr. Burns y Milhouse.

## Contenido del cuaderno

1. Carga de datos, división estratificada y caché en `.npy`
2. Red convolucional configurable (`ConvNet`)
3. Entrenamiento con parada anticipada (early stopping)
4. Búsqueda manual de hiperparámetros (C1–C7)
5. Aumentado de datos (C8–C9)
6. Búsqueda automática de hiperparámetros con Optuna (C10)
7. Transfer learning con ResNet18 (C11)
8. Comparativa global, selección del mejor modelo y evaluación en test

## Estructura

```
simpsons-cnn/
├── data/                dataset y cachés .npy (ignorado por git)
├── modelos/             pesos .pth de cada configuración (ignorado por git)
├── resultados/          gráficas, matrices de confusión y tablas de experimentos
├── informe/             informe final en PDF
├── simpsons_cnn.ipynb   cuaderno principal
└── requirements.txt
```

## Instalación y ejecución

```bash
pip install -r requirements.txt
```

Para usar GPU NVIDIA, instalar PyTorch con CUDA siguiendo https://pytorch.org/get-started/locally/

El cuaderno detecta si hay GPU. Sin GPU se ejecuta en modo prueba (100 imágenes por personaje a 64×64 y pocas épocas) para comprobar que el código funciona; los resultados definitivos se obtienen con GPU y el dataset completo (128×128).

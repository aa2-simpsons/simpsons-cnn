# simpsons-cnn

Clasificador de personajes de Los Simpson con redes neuronales convolucionales en PyTorch.

## Dataset

[The Simpsons Characters Data](https://www.kaggle.com/datasets/alexattia/the-simpsons-characters-dataset) (Kaggle).
Se descarga manualmente y se descomprime en la carpeta `data/` (no se sube al repositorio).

## Estructura

```
simpsons-cnn/
├── data/            dataset (ignorado por git)
├── resultados/      gráficas, matrices de confusión y registro de experimentos
├── informe/         informe final en PDF
├── simpsons_cnn.ipynb
└── requirements.txt
```

## Instalación

```bash
pip install -r requirements.txt
```

Para usar GPU NVIDIA, instalar PyTorch con CUDA siguiendo https://pytorch.org/get-started/locally/

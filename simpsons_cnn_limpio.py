import os
import random
from time import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import TensorDataset, DataLoader

SEMILLA = 42
random.seed(SEMILLA)
np.random.seed(SEMILLA)
torch.manual_seed(SEMILLA)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
MODO_PRUEBA = device == 'cpu'

print(f'Usando dispositivo: {device}')
print(f'Modo prueba: {MODO_PRUEBA}')

DIR_DATOS = 'data'
DIR_TRAIN = os.path.join(DIR_DATOS, 'simpsons_dataset')
DIR_TEST = os.path.join(DIR_DATOS, 'kaggle_simpson_testset', 'kaggle_simpson_testset')
DIR_RESULTADOS = 'resultados'
os.makedirs(DIR_RESULTADOS, exist_ok=True)

PERSONAJES = [
    'homer_simpson',
    'ned_flanders',
    'moe_szyslak',
    'lisa_simpson',
    'bart_simpson',
    'marge_simpson',
    'krusty_the_clown',
    'principal_skinner',
    'charles_montgomery_burns',
    'milhouse_van_houten',
]
NUM_CLASES = len(PERSONAJES)
NOMBRES = [p.replace('_', ' ').title() for p in PERSONAJES]

TAM_IMG = 64 if MODO_PRUEBA else 128
MAX_POR_CLASE = 100 if MODO_PRUEBA else None
PROP_VALIDACION = 0.15

def rutas_entrenamiento():
    rutas = []
    etiquetas = []
    for indice, personaje in enumerate(PERSONAJES):
        carpeta = os.path.join(DIR_TRAIN, personaje)
        archivos = sorted(f for f in os.listdir(carpeta) if f.endswith('.jpg'))
        if MAX_POR_CLASE:
            archivos = random.Random(SEMILLA).sample(archivos, min(MAX_POR_CLASE, len(archivos)))
        rutas += [os.path.join(carpeta, f) for f in archivos]
        etiquetas += [indice] * len(archivos)
    return rutas, etiquetas

def rutas_test():
    rutas = []
    etiquetas = []
    for archivo in sorted(os.listdir(DIR_TEST)):
        personaje = archivo.rsplit('_', 1)[0]
        if personaje in PERSONAJES:
            rutas.append(os.path.join(DIR_TEST, archivo))
            etiquetas.append(PERSONAJES.index(personaje))
    return rutas, etiquetas

rutas, etiquetas = rutas_entrenamiento()
rutas_train, rutas_val, etiquetas_train, etiquetas_val = train_test_split(
    rutas, etiquetas, test_size=PROP_VALIDACION, stratify=etiquetas, random_state=SEMILLA
)
rutas_test, etiquetas_test = rutas_test()

print(f'Entrenamiento: {len(rutas_train)}')
print(f'Validación: {len(rutas_val)}')
print(f'Test: {len(rutas_test)}')

conteo = pd.DataFrame({
    'Entrenamiento': pd.Series(etiquetas_train).value_counts(),
    'Validación': pd.Series(etiquetas_val).value_counts(),
    'Test': pd.Series(etiquetas_test).value_counts(),
}).sort_index()
conteo.index = NOMBRES
conteo.loc['TOTAL'] = conteo.sum()
conteo

ax = conteo.drop('TOTAL').plot(kind='barh', stacked=True, figsize=(9, 5))
ax.invert_yaxis()
plt.title('Imágenes por personaje y conjunto')
plt.xlabel('Número de imágenes')
plt.tight_layout()
plt.savefig(os.path.join(DIR_RESULTADOS, 'distribucion_clases.png'), dpi=150)
plt.show()

plt.figure(figsize=(15, 6))
for indice in range(NUM_CLASES):
    ruta = rutas_train[etiquetas_train.index(indice)]
    imagen = Image.open(ruta)
    plt.subplot(2, 5, indice + 1)
    plt.imshow(imagen)
    plt.title(f'{NOMBRES[indice]}\n{imagen.size[0]}x{imagen.size[1]}', fontsize=10)
    plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(DIR_RESULTADOS, 'ejemplos_personajes.png'), dpi=150)
plt.show()

def cargar_imagenes(rutas):
    imagenes = np.zeros((len(rutas), TAM_IMG, TAM_IMG, 3), dtype=np.uint8)
    for i, ruta in enumerate(rutas):
        imagen = Image.open(ruta).convert('RGB').resize((TAM_IMG, TAM_IMG))
        imagenes[i] = np.array(imagen)
    return imagenes

def cargar_conjunto(nombre, rutas, etiquetas):
    modo = 'prueba' if MODO_PRUEBA else 'completo'
    archivo_X = os.path.join(DIR_DATOS, f'{nombre}_{TAM_IMG}px_{modo}_X.npy')
    archivo_y = os.path.join(DIR_DATOS, f'{nombre}_{TAM_IMG}px_{modo}_y.npy')

    if not os.path.exists(archivo_X):
        comienzo = time()
        np.save(archivo_X, cargar_imagenes(rutas))
        np.save(archivo_y, np.array(etiquetas))
        print(f'{nombre}: {len(rutas)} imágenes leídas en {time() - comienzo:.1f} segundos')

    X = np.load(archivo_X)
    y = np.load(archivo_y)

    X = torch.from_numpy(X).permute(0, 3, 1, 2)
    X = X.float() / 255
    y = torch.from_numpy(y).long()
    return X, y

X_train, y_train = cargar_conjunto('train', rutas_train, etiquetas_train)
X_val, y_val = cargar_conjunto('val', rutas_val, etiquetas_val)
X_test, y_test = cargar_conjunto('test', rutas_test, etiquetas_test)

print(f'X_train: {X_train.shape}  y_train: {y_train.shape}')
print(f'X_val:   {X_val.shape}  y_val:   {y_val.shape}')
print(f'X_test:  {X_test.shape}  y_test:  {y_test.shape}')
print(f'Rango de valores: [{X_train.min():.1f}, {X_train.max():.1f}]')

dataset_train = TensorDataset(X_train, y_train)
dataset_val = TensorDataset(X_val, y_val)
dataset_test = TensorDataset(X_test, y_test)

def crear_dataloaders(tam_lote):
    loader_train = DataLoader(dataset_train, batch_size=tam_lote, shuffle=True)
    loader_val = DataLoader(dataset_val, batch_size=tam_lote, shuffle=False)
    loader_test = DataLoader(dataset_test, batch_size=tam_lote, shuffle=False)
    return loader_train, loader_val, loader_test

loader_train, loader_val, loader_test = crear_dataloaders(32)

imagenes, etiquetas = next(iter(loader_train))
print(f'Tamaño batch imágenes: {imagenes.shape}')
print(f'Tamaño batch etiquetas: {etiquetas.shape}')
print(f'Etiquetas: {etiquetas}')

plt.figure(figsize=(16, 4.5))
for i in range(16):
    plt.subplot(2, 8, i + 1)
    plt.imshow(imagenes[i].permute(1, 2, 0).numpy())
    plt.title(NOMBRES[etiquetas[i]], fontsize=8)
    plt.axis('off')
plt.suptitle(f'Lote de entrenamiento ({TAM_IMG}x{TAM_IMG})')
plt.tight_layout()
plt.show()

import torch.nn as nn
import torch.nn.functional as F

ACTIVACIONES = {
    'relu': F.relu,
    'leaky_relu': F.leaky_relu,
    'elu': F.elu,
    'tanh': torch.tanh,
}

class ConvNet(nn.Module):
    def __init__(self, filtros=(16, 32, 64), neuronas=(128,), activacion='relu', dropout=0.25, tam_kernel=5):
        super(ConvNet, self).__init__()
        self.activacion = ACTIVACIONES[activacion]

        self.convs = nn.ModuleList()
        canales_entrada = 3
        for canales_salida in filtros:
            self.convs.append(nn.Conv2d(in_channels=canales_entrada, out_channels=canales_salida,
                                        kernel_size=tam_kernel, stride=1, padding=tam_kernel // 2))
            canales_entrada = canales_salida

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(p=dropout)

        lado = TAM_IMG // 2 ** len(filtros)
        self.tam_aplanado = filtros[-1] * lado * lado

        self.fcs = nn.ModuleList()
        entradas = self.tam_aplanado
        for n in neuronas:
            self.fcs.append(nn.Linear(entradas, n))
            entradas = n
        self.salida = nn.Linear(entradas, NUM_CLASES)

    def forward(self, x):
        for conv in self.convs:
            x = self.pool(self.activacion(conv(x)))

        x = x.view(-1, self.tam_aplanado)

        for fc in self.fcs:
            x = self.dropout(self.activacion(fc(x)))

        return self.salida(x)

def resumen(modelo):
    x = torch.zeros(1, 3, TAM_IMG, TAM_IMG)
    print(f'{"Entrada":<28} {tuple(x.shape)}')
    for i, conv in enumerate(modelo.convs):
        x = modelo.pool(modelo.activacion(conv(x)))
        print(f'{f"Bloque conv {i + 1} ({conv.out_channels} filtros)":<28} {tuple(x.shape)}')
    x = x.view(-1, modelo.tam_aplanado)
    print(f'{"Aplanado":<28} {tuple(x.shape)}')
    for i, fc in enumerate(modelo.fcs):
        x = fc(x)
        print(f'{f"Capa FC {i + 1}":<28} {tuple(x.shape)}')
    x = modelo.salida(x)
    print(f'{"Salida":<28} {tuple(x.shape)}')
    total = sum(p.numel() for p in modelo.parameters())
    print(f'\nNúmero total de parámetros: {total:,}')

modelo = ConvNet()
print(modelo)
print()
resumen(modelo)

modelo = ConvNet().to(device)
imagenes, etiquetas = next(iter(loader_train))
salida = modelo(imagenes.to(device))
print(f'Entrada: {imagenes.shape}')
print(f'Salida: {salida.shape}')
print(f'Clases predichas (sin entrenar): {salida.argmax(dim=1)}')

import copy
import optuna
import torch.optim as optim
from sklearn.metrics import accuracy_score

def evaluar(modelo, loader, criterio):
    modelo.eval()
    perdida_total = 0.0
    y_real = []
    y_pred = []

    with torch.no_grad():
        for imagenes, etiquetas in loader:
            imagenes = imagenes.to(device)
            etiquetas = etiquetas.to(device)

            predicciones = modelo(imagenes)
            perdida = criterio(predicciones, etiquetas)
            perdida_total += perdida.item()

            y_pred.append(predicciones.argmax(dim=1).cpu().numpy())
            y_real.append(etiquetas.cpu().numpy())

    y_real = np.concatenate(y_real)
    y_pred = np.concatenate(y_pred)
    return perdida_total / len(loader), accuracy_score(y_real, y_pred), y_real, y_pred

def entrenar(modelo, loader_train, loader_val, lr=0.001, epocas=50, paciencia=5, aumento=None, trial=None):
    criterio = nn.CrossEntropyLoss()
    optimizador = optim.Adam(modelo.parameters(), lr=lr)

    historial = {'perdida_train': [], 'perdida_val': [], 'acc_train': [], 'acc_val': []}
    mejor_perdida_val = float('inf')
    mejores_pesos = None
    mejor_epoca = 0
    epocas_sin_mejora = 0

    comienzo = time()
    for epoca in range(epocas):
        modelo.train()
        perdida_total = 0.0
        y_real = []
        y_pred = []

        for imagenes, etiquetas in loader_train:
            imagenes = imagenes.to(device)
            etiquetas = etiquetas.to(device)
            if aumento is not None:
                imagenes = torch.stack([aumento(imagen) for imagen in imagenes])

            optimizador.zero_grad()
            predicciones = modelo(imagenes)
            perdida = criterio(predicciones, etiquetas)
            perdida.backward()
            optimizador.step()

            perdida_total += perdida.item()
            y_pred.append(predicciones.argmax(dim=1).detach().cpu().numpy())
            y_real.append(etiquetas.cpu().numpy())

        perdida_train = perdida_total / len(loader_train)
        acc_train = accuracy_score(np.concatenate(y_real), np.concatenate(y_pred))
        perdida_val, acc_val, _, _ = evaluar(modelo, loader_val, criterio)

        historial['perdida_train'].append(perdida_train)
        historial['perdida_val'].append(perdida_val)
        historial['acc_train'].append(acc_train)
        historial['acc_val'].append(acc_val)

        print(f'Época [{epoca + 1:2d}/{epocas}] | '
              f'Train Loss: {perdida_train:.4f} | Train Acc: {acc_train:.4f} | '
              f'Val Loss: {perdida_val:.4f} | Val Acc: {acc_val:.4f}')

        if trial is not None:
            trial.report(acc_val, epoca)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if perdida_val < mejor_perdida_val:
            mejor_perdida_val = perdida_val
            mejores_pesos = copy.deepcopy(modelo.state_dict())
            mejor_epoca = epoca + 1
            epocas_sin_mejora = 0
        else:
            epocas_sin_mejora += 1
            if epocas_sin_mejora >= paciencia:
                print(f'\nParada anticipada: {paciencia} épocas sin mejorar la pérdida de validación')
                break

    tiempo = time() - comienzo
    modelo.load_state_dict(mejores_pesos)
    print(f'\nMejor época: {mejor_epoca} | Val Loss: {mejor_perdida_val:.4f} | '
          f'Val Acc: {historial["acc_val"][mejor_epoca - 1]:.4f} | Tiempo: {tiempo:.1f} segundos')
    return historial, mejor_epoca, tiempo

def graficas(historial, mejor_epoca, nombre):
    epocas_eje = range(1, len(historial['perdida_train']) + 1)

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epocas_eje, historial['perdida_train'], label='Entrenamiento')
    plt.plot(epocas_eje, historial['perdida_val'], label='Validación')
    plt.axvline(mejor_epoca, color='gray', linestyle='--', label=f'Mejor época ({mejor_epoca})')
    plt.title('Evolución de la pérdida')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(epocas_eje, historial['acc_train'], label='Entrenamiento')
    plt.plot(epocas_eje, historial['acc_val'], label='Validación')
    plt.axvline(mejor_epoca, color='gray', linestyle='--', label=f'Mejor época ({mejor_epoca})')
    plt.title('Evolución de la precisión (accuracy)')
    plt.xlabel('Época')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.suptitle(nombre)
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_RESULTADOS, f'curvas_{nombre}.png'), dpi=150)
    plt.show()

DIR_MODELOS = 'modelos'
os.makedirs(DIR_MODELOS, exist_ok=True)

CONFIGURACIONES = [
    dict(nombre='C1_base', filtros=(16, 32, 64), neuronas=(128,), activacion='relu',
         dropout=0.25, tam_kernel=5, lr=0.001, tam_lote=32),
    dict(nombre='C2_mas_dropout', filtros=(16, 32, 64), neuronas=(128,), activacion='relu',
         dropout=0.5, tam_kernel=5, lr=0.001, tam_lote=32),
    dict(nombre='C3_mas_filtros', filtros=(32, 64, 128), neuronas=(256,), activacion='relu',
         dropout=0.5, tam_kernel=3, lr=0.001, tam_lote=32),
    dict(nombre='C4_4_capas_conv', filtros=(32, 64, 128, 256), neuronas=(256,), activacion='relu',
         dropout=0.5, tam_kernel=3, lr=0.001, tam_lote=32),
    dict(nombre='C5_lr_y_lote', filtros=(32, 64, 128, 256), neuronas=(256,), activacion='relu',
         dropout=0.5, tam_kernel=3, lr=0.0005, tam_lote=64),
    dict(nombre='C6_leaky_relu', filtros=(32, 64, 128, 256), neuronas=(256,), activacion='leaky_relu',
         dropout=0.5, tam_kernel=3, lr=0.0005, tam_lote=64),
    dict(nombre='C7_2_capas_fc', filtros=(32, 64, 128, 256), neuronas=(512, 128), activacion='relu',
         dropout=0.5, tam_kernel=3, lr=0.0005, tam_lote=64),
]

EPOCAS_MAX = 50
PACIENCIA = 5

def ejecutar_experimento(config, aumento=None):
    print('=' * 100)
    print(config, '| Data augmentation:', aumento is not None)
    print('=' * 100)

    torch.manual_seed(SEMILLA)
    loader_train, loader_val, loader_test = crear_dataloaders(config['tam_lote'])
    modelo = ConvNet(config['filtros'], config['neuronas'], config['activacion'],
                     config['dropout'], config['tam_kernel']).to(device)

    historial, mejor_epoca, tiempo = entrenar(modelo, loader_train, loader_val, lr=config['lr'],
                                              epocas=config.get('epocas', EPOCAS_MAX),
                                              paciencia=config.get('paciencia', PACIENCIA),
                                              aumento=aumento)
    graficas(historial, mejor_epoca, config['nombre'])
    torch.save(modelo.state_dict(), os.path.join(DIR_MODELOS, f"{config['nombre']}.pth"))

    return {
        'Configuración': config['nombre'],
        'Filtros': str(config['filtros']),
        'Neuronas FC': str(config['neuronas']),
        'Activación': config['activacion'],
        'Dropout': config['dropout'],
        'Kernel': config['tam_kernel'],
        'LR': config['lr'],
        'Lote': config['tam_lote'],
        'Aumento': 'Sí' if aumento is not None else 'No',
        'Parámetros': sum(p.numel() for p in modelo.parameters()),
        'Épocas': len(historial['perdida_train']),
        'Mejor época': mejor_epoca,
        'Train Acc': round(historial['acc_train'][mejor_epoca - 1], 4),
        'Val Loss': round(historial['perdida_val'][mejor_epoca - 1], 4),
        'Val Acc': round(historial['acc_val'][mejor_epoca - 1], 4),
        'Tiempo (s)': round(tiempo, 1),
    }

resultados = []
for config in CONFIGURACIONES:
    resultados.append(ejecutar_experimento(config))
    pd.DataFrame(resultados).to_csv(os.path.join(DIR_RESULTADOS, 'experimentos.csv'), index=False)

tabla = pd.DataFrame(resultados)
tabla

plt.figure(figsize=(10, 5))
barras = plt.bar(tabla['Configuración'], tabla['Val Acc'])
plt.bar_label(barras, labels=[f'{v:.1%}' for v in tabla['Val Acc']])
plt.axhline(0.85, color='red', linestyle='--', label='Objetivo 85%')
plt.ylim(0.5, 1)
plt.title('Accuracy de validación por configuración')
plt.ylabel('Accuracy de validación')
plt.xticks(rotation=30, ha='right')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(DIR_RESULTADOS, 'comparativa_configuraciones.png'), dpi=150)
plt.show()

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

def matriz_confusion(y_real, y_pred, titulo, archivo):
    matriz = confusion_matrix(y_real, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay(matriz, display_labels=NOMBRES).plot(ax=ax, cmap='Blues', xticks_rotation=45, colorbar=False)
    plt.title(titulo)
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_RESULTADOS, archivo), dpi=150)
    plt.show()

mejor = tabla.loc[tabla['Val Acc'].idxmax(), 'Configuración']
config = next(c for c in CONFIGURACIONES if c['nombre'] == mejor)
print(f'Mejor configuración: {mejor}')

modelo = ConvNet(config['filtros'], config['neuronas'], config['activacion'],
                 config['dropout'], config['tam_kernel']).to(device)
modelo.load_state_dict(torch.load(os.path.join(DIR_MODELOS, f'{mejor}.pth')))

loader_train, loader_val, loader_test = crear_dataloaders(config['tam_lote'])
perdida_val, acc_val, y_real_val, y_pred_val = evaluar(modelo, loader_val, nn.CrossEntropyLoss())
print(f'Val Loss: {perdida_val:.4f} | Val Acc: {acc_val:.4f}\n')
print(classification_report(y_real_val, y_pred_val, target_names=NOMBRES, digits=3))

matriz_confusion(y_real_val, y_pred_val, f'Matriz de confusión – validación ({mejor})', f'matriz_confusion_val_{mejor}.png')

from torchvision import transforms

aumento = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.RandomResizedCrop(size=TAM_IMG, scale=(0.8, 1.0), antialias=True),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
])

imagen = X_train[0]
plt.figure(figsize=(16, 4.5))
plt.subplot(2, 8, 1)
plt.imshow(imagen.permute(1, 2, 0).numpy())
plt.title('Original', fontsize=9)
plt.axis('off')
for i in range(2, 17):
    plt.subplot(2, 8, i)
    plt.imshow(aumento(imagen).clamp(0, 1).permute(1, 2, 0).numpy())
    plt.title(f'Aumentada {i - 1}', fontsize=9)
    plt.axis('off')
plt.suptitle(f'Ejemplos de data augmentation ({NOMBRES[y_train[0]]})')
plt.tight_layout()
plt.savefig(os.path.join(DIR_RESULTADOS, 'ejemplos_aumento.png'), dpi=150)
plt.show()

CONFIGURACIONES_AUMENTO = [
    dict(nombre='C8_C6_con_aumento', filtros=(32, 64, 128, 256), neuronas=(256,), activacion='leaky_relu',
         dropout=0.5, tam_kernel=3, lr=0.0005, tam_lote=64, epocas=80, paciencia=8),
    dict(nombre='C9_C7_con_aumento', filtros=(32, 64, 128, 256), neuronas=(512, 128), activacion='relu',
         dropout=0.5, tam_kernel=3, lr=0.0005, tam_lote=64, epocas=80, paciencia=8),
]

for config in CONFIGURACIONES_AUMENTO:
    resultados.append(ejecutar_experimento(config, aumento=aumento))
    pd.DataFrame(resultados).to_csv(os.path.join(DIR_RESULTADOS, 'experimentos.csv'), index=False)

tabla = pd.DataFrame(resultados)
tabla

plt.figure(figsize=(11, 5))
colores = ['tab:orange' if a == 'Sí' else 'tab:blue' for a in tabla['Aumento']]
barras = plt.bar(tabla['Configuración'], tabla['Val Acc'], color=colores)
plt.bar_label(barras, labels=[f'{v:.1%}' for v in tabla['Val Acc']])
plt.axhline(0.85, color='red', linestyle='--', label='Objetivo 85%')
plt.bar(0, 0, color='tab:blue', label='Sin aumento')
plt.bar(0, 0, color='tab:orange', label='Con aumento')
plt.ylim(0.5, 1)
plt.title('Accuracy de validación por configuración')
plt.ylabel('Accuracy de validación')
plt.xticks(rotation=30, ha='right')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(DIR_RESULTADOS, 'comparativa_aumento.png'), dpi=150)
plt.show()

mejor_aumento = max(CONFIGURACIONES_AUMENTO, key=lambda c: tabla.set_index('Configuración').loc[c['nombre'], 'Val Acc'])
print(f"Mejor configuración con aumento: {mejor_aumento['nombre']}")

modelo = ConvNet(mejor_aumento['filtros'], mejor_aumento['neuronas'], mejor_aumento['activacion'],
                 mejor_aumento['dropout'], mejor_aumento['tam_kernel']).to(device)
modelo.load_state_dict(torch.load(os.path.join(DIR_MODELOS, f"{mejor_aumento['nombre']}.pth")))

loader_train, loader_val, loader_test = crear_dataloaders(mejor_aumento['tam_lote'])
perdida_val, acc_val, y_real_val, y_pred_val = evaluar(modelo, loader_val, nn.CrossEntropyLoss())
print(f'Val Loss: {perdida_val:.4f} | Val Acc: {acc_val:.4f}\n')
print(classification_report(y_real_val, y_pred_val, target_names=NOMBRES, digits=3))
matriz_confusion(y_real_val, y_pred_val, f"Matriz de confusión – validación ({mejor_aumento['nombre']})",
                 f"matriz_confusion_val_{mejor_aumento['nombre']}.png")

optuna.logging.set_verbosity(optuna.logging.WARNING)

N_PRUEBAS = 5 if MODO_PRUEBA else 25
EPOCAS_BUSQUEDA = 3 if MODO_PRUEBA else 25
PACIENCIA_BUSQUEDA = 4

def objetivo(trial):
    capas_conv = trial.suggest_int('capas_conv', 3, 4)
    filtros_base = trial.suggest_categorical('filtros_base', [16, 32, 64])
    tam_kernel = trial.suggest_categorical('tam_kernel', [3, 5])
    capas_fc = trial.suggest_int('capas_fc', 1, 2)
    neuronas_base = trial.suggest_categorical('neuronas_base', [128, 256, 512])
    activacion = trial.suggest_categorical('activacion', ['relu', 'leaky_relu', 'elu'])
    dropout = trial.suggest_float('dropout', 0.2, 0.6)
    lr = trial.suggest_float('lr', 1e-4, 3e-3, log=True)
    tam_lote = trial.suggest_categorical('tam_lote', [32, 64, 128])

    filtros = tuple(filtros_base * 2 ** i for i in range(capas_conv))
    neuronas = tuple(neuronas_base // 2 ** i for i in range(capas_fc))

    torch.manual_seed(SEMILLA)
    loader_train, loader_val, loader_test = crear_dataloaders(tam_lote)
    modelo = ConvNet(filtros, neuronas, activacion, dropout, tam_kernel).to(device)

    print(f'Prueba {trial.number}: filtros={filtros}, neuronas={neuronas}, {activacion}, '
          f'dropout={dropout:.2f}, kernel={tam_kernel}, lr={lr:.5f}, lote={tam_lote}')
    historial, mejor_epoca, tiempo = entrenar(modelo, loader_train, loader_val, lr=lr,
                                              epocas=EPOCAS_BUSQUEDA, paciencia=PACIENCIA_BUSQUEDA,
                                              trial=trial)
    return historial['acc_val'][mejor_epoca - 1]

estudio = optuna.create_study(direction='maximize',
                              sampler=optuna.samplers.TPESampler(seed=SEMILLA),
                              pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3))
comienzo = time()
estudio.optimize(objetivo, n_trials=N_PRUEBAS)
print(f'\nBúsqueda completada en {(time() - comienzo) / 60:.1f} minutos')

pruebas = estudio.trials_dataframe(attrs=('number', 'value', 'state', 'params', 'duration'))
pruebas['duration'] = pruebas['duration'].dt.total_seconds().round(1)
pruebas = pruebas.rename(columns={'number': 'Prueba', 'value': 'Val Acc', 'state': 'Estado', 'duration': 'Tiempo (s)'})
pruebas.columns = [c.replace('params_', '') for c in pruebas.columns]
pruebas.to_csv(os.path.join(DIR_RESULTADOS, 'optuna_pruebas.csv'), index=False)

print(f'Pruebas completas: {(pruebas["Estado"] == "COMPLETE").sum()} | Podadas: {(pruebas["Estado"] == "PRUNED").sum()}')
print(f'Mejor Val Acc: {estudio.best_value:.4f}')
print(f'Mejores hiperparámetros: {estudio.best_params}')
pruebas.sort_values('Val Acc', ascending=False)

plt.figure(figsize=(10, 5))
completas = pruebas[pruebas['Estado'] == 'COMPLETE']
podadas = pruebas[pruebas['Estado'] == 'PRUNED']
plt.scatter(completas['Prueba'], completas['Val Acc'], label='Completada')
plt.scatter(podadas['Prueba'], podadas['Val Acc'], marker='x', color='gray', label='Podada')
plt.plot(completas['Prueba'], completas['Val Acc'].cummax(), color='red', label='Mejor hasta el momento')
plt.title('Historial de la búsqueda automática (Optuna)')
plt.xlabel('Número de prueba')
plt.ylabel('Accuracy de validación')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(DIR_RESULTADOS, 'optuna_historial.png'), dpi=150)
plt.show()

importancias = optuna.importance.get_param_importances(estudio)
plt.figure(figsize=(8, 4))
plt.barh(list(importancias.keys()), list(importancias.values()))
plt.gca().invert_yaxis()
plt.title('Importancia de cada hiperparámetro (Optuna)')
plt.xlabel('Importancia')
plt.tight_layout()
plt.savefig(os.path.join(DIR_RESULTADOS, 'optuna_importancias.png'), dpi=150)
plt.show()

p = estudio.best_params
config_optuna = dict(
    nombre='C10_optuna_con_aumento',
    filtros=tuple(p['filtros_base'] * 2 ** i for i in range(p['capas_conv'])),
    neuronas=tuple(p['neuronas_base'] // 2 ** i for i in range(p['capas_fc'])),
    activacion=p['activacion'], dropout=round(p['dropout'], 2), tam_kernel=p['tam_kernel'],
    lr=p['lr'], tam_lote=p['tam_lote'],
    epocas=3 if MODO_PRUEBA else 80, paciencia=8,
)
resultados.append(ejecutar_experimento(config_optuna, aumento=aumento))
pd.DataFrame(resultados).to_csv(os.path.join(DIR_RESULTADOS, 'experimentos.csv'), index=False)
tabla = pd.DataFrame(resultados)
tabla

import torchvision.models as models

class ResNetSimpsons(nn.Module):
    def __init__(self, dropout=0.5, preentrenada=True):
        super(ResNetSimpsons, self).__init__()
        self.normalizar = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        pesos = models.ResNet18_Weights.DEFAULT if preentrenada else None
        self.red = models.resnet18(weights=pesos)
        for param in self.red.parameters():
            param.requires_grad = False
        for param in self.red.layer4.parameters():
            param.requires_grad = True
        self.red.fc = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(self.red.fc.in_features, NUM_CLASES))

    def forward(self, x):
        return self.red(self.normalizar(x))

torch.manual_seed(SEMILLA)
loader_train, loader_val, loader_test = crear_dataloaders(64)
modelo_resnet = ResNetSimpsons().to(device)

parametros_totales = sum(p.numel() for p in modelo_resnet.parameters())
parametros_entrenables = sum(p.numel() for p in modelo_resnet.parameters() if p.requires_grad)
print(f'Parámetros totales: {parametros_totales:,} | Entrenables: {parametros_entrenables:,}')

historial_resnet, mejor_epoca_resnet, tiempo_resnet = entrenar(
    modelo_resnet, loader_train, loader_val, lr=0.0005,
    epocas=3 if MODO_PRUEBA else 40, paciencia=6, aumento=aumento
)
graficas(historial_resnet, mejor_epoca_resnet, 'C11_resnet18_finetuning')
torch.save(modelo_resnet.state_dict(), os.path.join(DIR_MODELOS, 'C11_resnet18_finetuning.pth'))

resultados.append({
    'Configuración': 'C11_resnet18_finetuning',
    'Filtros': 'ResNet18 preentrenada (layer4 + fc entrenables)',
    'Neuronas FC': f'({modelo_resnet.red.fc[1].in_features}, {NUM_CLASES})',
    'Activación': 'relu',
    'Dropout': 0.5,
    'Kernel': '3x3 / 7x7',
    'LR': 0.0005,
    'Lote': 64,
    'Aumento': 'Sí',
    'Parámetros': parametros_totales,
    'Épocas': len(historial_resnet['perdida_train']),
    'Mejor época': mejor_epoca_resnet,
    'Train Acc': round(historial_resnet['acc_train'][mejor_epoca_resnet - 1], 4),
    'Val Loss': round(historial_resnet['perdida_val'][mejor_epoca_resnet - 1], 4),
    'Val Acc': round(historial_resnet['acc_val'][mejor_epoca_resnet - 1], 4),
    'Tiempo (s)': round(tiempo_resnet, 1),
})
pd.DataFrame(resultados).to_csv(os.path.join(DIR_RESULTADOS, 'experimentos.csv'), index=False)
pd.DataFrame(resultados)

tabla_global = pd.DataFrame(resultados)

plt.figure(figsize=(14, 6))
colores = []
for conf in tabla_global["Configuración"]:
    if "optuna" in conf:
        colores.append("tab:green")
    elif "transfer" in conf:
        colores.append("tab:purple")
    elif "con_aumento" in conf:
        colores.append("tab:orange")
    else:
        colores.append("tab:blue")

barras = plt.bar(tabla_global["Configuración"], tabla_global["Val Acc"], color=colores)
plt.bar_label(barras, labels=[f"{v:.1%}" for v in tabla_global["Val Acc"]], padding=3, fontsize=9)
plt.axhline(0.85, color="red", linestyle="--", linewidth=1.5, label="Objetivo mínimo de la práctica (85%)")
plt.xticks(rotation=45, ha="right", fontsize=10)
plt.ylabel("Exactitud en validación (Val Acc)")
plt.title("Comparativa global de todas las configuraciones evaluadas")
plt.ylim(0, 1.05)
plt.grid(axis="y", linestyle=":", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(DIR_RESULTADOS, "comparativa_final.png"), dpi=150)
plt.show()

mejor_idx = tabla_global["Val Loss"].idxmin()
mejor_cfg = tabla_global.loc[mejor_idx]
print("=" * 80)
print(f"MEJOR MODELO SELECCIONADO: {mejor_cfg['Configuración']}")
print(f"Val Acc: {mejor_cfg['Val Acc']:.2%} | Val Loss: {mejor_cfg['Val Loss']:.4f} | Época óptima: {mejor_cfg['Mejor época']}")
print("=" * 80)

nombre_mejor = mejor_cfg['Configuración']
ruta_pesos = os.path.join(DIR_MODELOS, f'{nombre_mejor}.pth')

if 'resnet' in nombre_mejor:
    modelo_final = ResNetSimpsons(preentrenada=False)
else:
    todas_las_configs = CONFIGURACIONES + CONFIGURACIONES_AUMENTO + [config_optuna]
    cfg_sel = next(c for c in todas_las_configs if c['nombre'] == nombre_mejor)
    modelo_final = ConvNet(cfg_sel['filtros'], cfg_sel['neuronas'], cfg_sel['activacion'],
                           cfg_sel['dropout'], cfg_sel['tam_kernel'])
modelo_final.load_state_dict(torch.load(ruta_pesos, map_location=device))
modelo_final = modelo_final.to(device)

perdida_test, acc_test, y_test_real, y_test_pred = evaluar(modelo_final, loader_test, nn.CrossEntropyLoss())

print('=' * 80)
print(f'EVALUACIÓN DEFINITIVA EN EL CONJUNTO DE TEST ({nombre_mejor})')
print(f'Exactitud en test: {acc_test:.2%}  (objetivo: >= 85%)')
print(f'Pérdida en test:   {perdida_test:.4f}')
print('=' * 80)
print()
print(classification_report(y_test_real, y_test_pred, target_names=NOMBRES, digits=3))
matriz_confusion(y_test_real, y_test_pred, f'Matriz de confusión – test ({nombre_mejor}, {acc_test:.1%})', 'matriz_confusion_test.png')

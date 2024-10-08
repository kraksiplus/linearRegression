import os
import glob

# Inicializar diccionarios para almacenar las sumas y conteos de cada métrica
metrics_sum = {
    'mae': 0.0,
    'mse': 0.0,
    'medae': 0.0,
    'msle': 0.0,
    'rmsle': 0.0,
    'evs': 0.0,
    'r2': 0.0
}
metrics_count = {
    'mae': 0,
    'mse': 0,
    'medae': 0,
    'msle': 0,
    'rmsle': 0,
    'evs': 0,
    'r2': 0
}

# Obtener todos los archivos .txt en el directorio actual
txt_files = glob.glob("*.txt")

# Leer cada archivo y acumular las métricas
for file in txt_files:
    with open(file, 'r') as f:
        for line in f:
            key, value = line.split(':')
            key = key.strip()
            value = float(value.strip())
            if key in metrics_sum:
                metrics_sum[key] += value
                metrics_count[key] += 1

# Calcular los promedios
metrics_avg = {key: metrics_sum[key] / metrics_count[key] for key in metrics_sum}

# Imprimir los promedios
for key, avg in metrics_avg.items():
    print(f"{key}: {avg}")
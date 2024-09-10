# Proyecto de Regresión Lineal con Keras

Este proyecto implementa un modelo de regresión lineal utilizando la biblioteca Keras en Python. El modelo predice el peso de una persona en función de su altura utilizando un conjunto de datos proporcionado en un archivo CSV.

## Requisitos

Asegúrate de tener los siguientes paquetes instalados en tu entorno de Python:

- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `tensorflow` (incluye Keras)

Puedes instalar todos estos paquetes con el siguiente comando:

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow
```
# Archivos del Proyecto
main.py: Script principal que lee los datos, entrena el modelo de regresión lineal, genera gráficos y realiza predicciones.
altura_peso.csv: Archivo CSV con los datos de altura y peso.

# Instrucciones de Uso
Preparar el Archivo CSV

Asegúrate de que el archivo altura_peso.csv esté en el mismo directorio que main.py. El archivo debe tener dos columnas sin encabezados: una para la altura y otra para el peso.

# Ejecutar el Script

Abre una terminal, navega al directorio del proyecto y asegúrate de que tu entorno virtual esté activado. Luego, ejecuta el script con:

```bash
python main.py

```

# Resultados

El script entrenará un modelo de regresión lineal con los datos del archivo CSV.
Generará dos gráficos: uno mostrando la evolución de la pérdida (ECM) durante el entrenamiento y otro mostrando la regresión lineal ajustada a los datos originales.
Imprimirá la predicción del peso para una persona de 170 cm de altura.

# Detalles del Modelo
Modelo: Red neuronal secuencial con una capa densa.
Función de pérdida: Error cuadrático medio (MSE).
Optimizador: Descenso de gradiente estocástico (SGD) con una tasa de aprendizaje ajustable.

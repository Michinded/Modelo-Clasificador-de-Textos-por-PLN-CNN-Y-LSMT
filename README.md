# Modelo Clasificador de Textos por Procesamiento de Lenguaje Natural (PLN) usando Redes Neuronales Convolucionales (CNN) y Long Short-Term Memory (LSTM)

## Descripción
Este repositorio contiene un modelo de aprendizaje automático para clasificar textos según sus autores utilizando técnicas avanzadas de Procesamiento de Lenguaje Natural (PLN) y redes neuronales convolucionales (CNN) combinadas con unidades de memoria a corto y largo plazo (LSTM). El objetivo es implementar un enfoque académico y profesional para la clasificación de textos.

## Contenido
- [Requisitos](#requisitos)
- [Uso](#uso)
- [Estructura del Código](#estructura-del-código)
- [Agradecimientos](#agradecimientos)

## Requisitos
- Python 3.x
- TensorFlow
- Keras
- pandas
- scikit-learn
- matplotlib
- seaborn

## Uso
1. Asegúrate de tener los requisitos instalados en tu entorno.
2. Coloca tu archivo de conjunto de datos ('data_set.csv') y archivo de nuevos textos ('nuevos_textos.csv') en el mismo directorio que el código.
3. Ejecuta el código y sigue las instrucciones para entrenar el modelo y realizar predicciones.

## Estructura del Código
- **Lectura del Dataset:** Inicio del código con la lectura del conjunto de datos que contiene los textos y los autores correspondientes.
- **División del Dataset:** División del conjunto de datos en conjuntos de entrenamiento y prueba.
- **Tokenización de los Textos:** Tokenización de los textos utilizando la clase `Tokenizer` de Keras.
- **Creación y Compilación del Modelo:** Construcción del modelo con capas de incrustación, capas convolucionales 1D, capas LSTM y capas densas.
- **Entrenamiento del Modelo:** Entrenamiento del modelo con monitorización de la métrica 'val_accuracy' utilizando Early Stopping.
- **Evaluación del Modelo:** Evaluación del modelo en el conjunto de prueba para calcular la pérdida y la precisión.
- **Predicción de Nuevos Textos:** Utilización del modelo para predecir los autores de nuevos textos proporcionados en un archivo CSV.
- **Resultados de la Predicción:** Almacenamiento de los resultados de la predicción en un DataFrame y cálculo de la precisión del modelo en los nuevos textos.
- **Gráficas y Visualización:** Generación de gráficas para visualizar las curvas de pérdida y precisión, así como una matriz de confusión.
- **Guardado de Resultados:** Guardado de los resultados de la predicción en un archivo CSV.

## Agradecimientos
Este código fue desarrollado por JUAN MIGUEL ROJAS ROMERO y FRANCISCO ANTONIO CASTILLO VELASQUEZ como parte de Verano de la Ciencia de la Region Centro (https://veranoregional.org/appVerano/). Si utilizas este código o te inspiras en él, por favor, considera dar crédito a los autores y proporcionar un enlace a este repositorio. ¡Esperamos que encuentres útil esta implementación de PLN y redes neuronales para clasificación de textos!



import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
# import para Regularización del modelo
from keras.regularizers import l1, l2
from sklearn.metrics import confusion_matrix
import seaborn as sns

"""
Lectura del dataset
"""

df = pd.read_csv('data_set.csv')

textos = df['datos'].tolist() # Lista de textos

autores = df['autores'].tolist() # Lista de autores correspondientes a los textos

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(autores)
print("Clasificando autores")

""""
 División del dataset en entrenamiento y prueba
"""

x_train, x_test, y_train, y_test = train_test_split(textos, y, test_size=0.2, random_state=42)
print("Datos divididos")

"""
 Tokenización de los textos (10,000 palabras)
"""

tokenizer = keras.preprocessing.text.Tokenizer(num_words=10000)
tokenizer.fit_on_texts(x_train)
print("Datos tokenizados")

x_train_seq = tokenizer.texts_to_sequences(x_train)
x_test_seq = tokenizer.texts_to_sequences(x_test)
print("Datos secuenciados")

max_seq_length = max(len(seq) for seq in x_train_seq)
x_train_seq = keras.preprocessing.sequence.pad_sequences(x_train_seq, maxlen=max_seq_length)
x_test_seq = keras.preprocessing.sequence.pad_sequences(x_test_seq, maxlen=max_seq_length)
print("Datos secuenciados y paddeados")

"""
 Creación y compilación del modelo
"""
model = keras.Sequential([
    keras.layers.Embedding(input_dim=10000, output_dim=64, input_length=max_seq_length),
    keras.layers.Conv1D(128, 5, activation='relu'),
    keras.layers.LSTM(64),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(len(label_encoder.classes_), activation='softmax',
                       kernel_regularizer=l2(0.001),
                       activity_regularizer=l1(0.001))
])
print("Modelo construido")

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print("Modelo compilado")

""" Entrenamiento del modelo. Este estará monitoreando el valor de "val_accuracy" para saber si este mejora en un 4% con una paciencia de 25 epocas.
"""
# Crea un objeto EarlyStopping
early_stopping = EarlyStopping(monitor='val_accuracy', min_delta=0.04, patience=25)
# Entrena el modelo con EarlyStopping
history = model.fit(x_train_seq, y_train, epochs=100, batch_size=32, validation_data=(x_test_seq, y_test),
                    callbacks=[early_stopping])
# Imprime un mensaje cuando el entrenamiento se detiene
print("Entrenamiento detenido por EarlyStopping")

"""
Guarda el modelo entrenado para su uso posterior
"""

model.save('modelo.h5')

"""
Gráficas de entrenamiento Accuracy y Loss
"""

# Trazar las curvas de perdida y precisión
print("Trazamiento de las curvas de pérdida y precisión.")
train_loss = history.history['loss']
train_accuracy = history.history['accuracy']
val_loss = history.history['val_loss']
val_accuracy = history.history['val_accuracy']

# Trazar las curvas de perdida
print("Gráfica de perdida: ")
plt.plot(train_loss, label='Pérdida de entrenamiento')
plt.plot(val_loss, label='Pérdida de validación')
plt.xlabel('Epocas')
plt.ylabel('Perdida')
plt.title("Gráfica de perdida")
plt.legend()
plt.show()

# Trazar las curvas de precisión
print("\n\nGráfica de precisión: ")
plt.plot(train_accuracy, label='Precisión de entrenamiento')
plt.plot(val_accuracy, label='Precisión de validación')
plt.xlabel('Epocas')
plt.ylabel('Precisión')
plt.title("Gráfica de precisión")
plt.legend()
plt.show()

"""
Evaluación del modelo
"""

loss, accuracy = model.evaluate(x_test_seq, y_test)
print('Perdida:', loss)
print('Precisión:', accuracy)

"""
Predicción de textos en el modelo usando un archivo de comas separadas (csv)
"""

# Lee el archivo CSV que contiene los nuevos textos
nuevos_df = pd.read_csv('nuevos_textos.csv')
# Obtiene los textos de los nuevos textos
nuevos_textos = nuevos_df['nuevos'].tolist()
# Tokeniza y secuencia los nuevos textos
nuevos_textos_seq = tokenizer.texts_to_sequences(nuevos_textos)
nuevos_textos_seq = keras.preprocessing.sequence.pad_sequences(nuevos_textos_seq, maxlen=max_seq_length)

"""
Resultados de la predicción
"""

# Predice los autores de los nuevos textos
predictions_nuevos = model.predict(nuevos_textos_seq)
predicted_authors_nuevos = label_encoder.inverse_transform(np.argmax(predictions_nuevos, axis=1))
contador = 0
for texto, author in zip(nuevos_textos, predicted_authors_nuevos):
    contador += 1
    print(f'Texto: "{contador}" | Autor probable: {author}')

"""
Obtener los resultados de la predicción en un archivo csv
"""

# Leer el archivo CSV que contiene los nuevos textos
nuevos_df = pd.read_csv('nuevos_textos.csv')

# Obtener los autores reales de los nuevos textos desde el archivo CSV
autores_reales_nuevos = nuevos_df['autores'].tolist()

# Comparar los autores reales con los autores predichos para los nuevos textos
aciertos_nuevos = [y_pred == autor_real for y_pred, autor_real in zip(predicted_authors_nuevos, autores_reales_nuevos)]

# Calcular la cantidad de textos en los nuevos datos
cantidad_textos_nuevos = len(aciertos_nuevos)

# Calcular la cantidad de aciertos en los nuevos textos
cantidad_aciertos_nuevos = sum(aciertos_nuevos)

# Calcular la precisión del modelo en los nuevos textos
precision_nuevos = cantidad_aciertos_nuevos / cantidad_textos_nuevos

# Expresar la precisión en porcentaje con dos decimales
precision_porcentaje = precision_nuevos * 100

print(f'\nCantidad de textos en los nuevos datos: {cantidad_textos_nuevos}')
print(f'Cantidad de aciertos en los nuevos textos: {cantidad_aciertos_nuevos}')
print(f'Precisión del modelo en los nuevos textos: {precision_porcentaje:.2f}%')

"""
Guarda los resultados de la predicción en un archivo csv
"""

# Crear un DataFrame con los resultados y los autores reales y probables, así como la cantidad de textos, cantidad de aciertos y precisión
resultados_df = pd.DataFrame({
    'Texto': nuevos_df['nuevos'],
    'Autor Real': autores_reales_nuevos,
    'Autor Probable': predicted_authors_nuevos
})
# Guardar los resultados en un nuevo archivo CSV
resultados_df.to_csv('resultados_modelo.csv', index=False)

"""
Compara los resultados de la predicción con los autores reales
"""

# Leer el archivo 'resultados_modelo.csv'
resultados_df = pd.read_csv('resultados_modelo.csv')

# Comparar las columnas 'Autor Real' y 'Autor Probable'
resultados_df['Aciertos'] = resultados_df['Autor Real'] == resultados_df['Autor Probable']

# Convertir el valor booleano True/False a 1/0
resultados_df['Aciertos'] = resultados_df['Aciertos'].astype(int)

# Guardar los resultados actualizados en un nuevo archivo CSV
resultados_df.to_csv('resultados_modelo.csv', index=False)

"""
Crea una gráfica de barras con los resultados de la predicción
"""

# Leer el archivo 'resultados_modelo.csv'
resultados_df = pd.read_csv('resultados_modelo.csv')

# Comparar las columnas 'Autor Real' y 'Autor Probable'
resultados_df['Aciertos'] = resultados_df['Autor Real'] == resultados_df['Autor Probable']

# Convertir el valor booleano True/False a 1/0
resultados_df['Aciertos'] = resultados_df['Aciertos'].astype(int)

# Calcular la cantidad de aciertos y diferencias
cantidad_aciertos = resultados_df['Aciertos'].sum()
cantidad_diferencias = len(resultados_df) - cantidad_aciertos

# Calcular el total de textos
total_textos = len(resultados_df)

# Crear una gráfica de barras
etiquetas = ['Aciertos', 'Diferencias']
valores = [cantidad_aciertos, cantidad_diferencias]
plt.bar(etiquetas, valores, color=['green', 'red'])

# Agregar etiquetas a los ejes
plt.xlabel('Resultados')
plt.ylabel('Cantidad')
plt.title('Resultados del Modelo')

# Mostrar el total de textos en el eje y
plt.ylim(0, total_textos)

# Agregar el valor del total de textos como una etiqueta en la gráfica
for i, valor in enumerate(valores):
    plt.text(i, valor, str(valor), ha='center', va='bottom')

# Mostrar la gráfica
plt.show()

"""
Crea una grafica de la matriz de confusión
"""

# Leer el archivo 'resultados_modelo.csv'
resultados_df = pd.read_csv('resultados_modelo.csv')

# Crear una matriz de confusión
matriz_confusion = confusion_matrix(resultados_df['Autor Real'], resultados_df['Autor Probable'])

# Crear una gráfica de matriz de confusión
sns.heatmap(matriz_confusion, annot=True, fmt='d', cmap='Blues')

# Agregar etiquetas a los ejes
plt.xlabel('Autor Probable')
plt.ylabel('Autor Real')
plt.title('Matriz de Confusión')

# Mostrar la gráfica
plt.show()
#importaciones
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

#lectura del csv
datos = pd.read_csv('altura_peso.csv', sep=",", header=None, names=['Altura', 'Peso'])
print(datos)
x = datos['Altura'].values
y = datos['Peso'].values

# Normalización de datos
scaler_x = StandardScaler()
scaler_y = StandardScaler()

x = scaler_x.fit_transform(x.reshape(-1, 1))
y = scaler_y.fit_transform(y.reshape(-1, 1))

#implementacion del modelo
np.random.seed(2)
modelo = Sequential()
modelo.add(Dense(1, input_dim=1, activation='linear'))

sgd = SGD(learning_rate=0.0004)  # Ajusta el learning rate según sea necesario
modelo.compile(loss='mse', optimizer=sgd)

#entrenamiento del modelo
num_epochs = 10000
batch_size = x.shape[0]
historia = modelo.fit(x, y, epochs=num_epochs, batch_size=batch_size, verbose=1)

#implementacion de graficos
plt.subplot(1,2,1)
plt.plot(historia.history['loss'])
plt.xlabel('epoch')
plt.ylabel('ECM')
plt.title('ECM vs. epochs')
y_regr = modelo.predict(x)
plt.subplot(1, 2, 2)
plt.scatter(scaler_x.inverse_transform(x), scaler_y.inverse_transform(y))  # Datos originales
plt.plot(scaler_x.inverse_transform(x), scaler_y.inverse_transform(y_regr), 'r')  # Regresión lineal
plt.title('Datos originales y regresión lineal')
plt.show()

#prediccion requerida
x_pred = np.array([[170]])
x_pred_scaled = scaler_x.transform(x_pred)
y_pred_scaled = modelo.predict(x_pred_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
print("El peso será de {:.1f} kg para una persona de {} cm".format(y_pred[0][0], x_pred[0][0]))

# Parkinson
Data set Parkinson Thesis 



import numpy as np
import pandas as pd

import missingno as msno
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import seaborn as sb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import plot_roc_curve
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

pip install tensorflow

"""# Parkinsons
Se manda a llamar la base de datos de parkinsons.csv
"""

df = pd.read_csv("parkinsons.csv")

"""# Capitulo II

#Visualizacion de los datos

Se manda llamar la base de datos con la estructura data frames, donde se observan los valores reales y estados(Status), que serán los que se manejararemos.
"""

df.head()

"""Se crea el codigo, apartir de mi base de datos de parkinson.csv

todos los valores los marcan en flotantes y estados (status) es int, ya que es el valor de interes y será el que se ocuapara para los codigos
"""

df.info()

"""Se obtiene todos los datos de la base de datos para su visualización.

Se realiza un analicis exploratorio sobre los codigos para ver que ninguno falte en y esten en total de 195, el valor en morado spread 1 y 2 maneja ambos.

# Summary
"""

str(df)

"""# Medición de dispercion"""

df = pd.DataFrame({"MDVP:Fo(Hz) ":[119.992, 122.400, 116.682, 116.676, 116.014],
                   "MDVP:Fhi(Hz) ":[157.302, 148.650, 131.111, 137.871, 141.781],
                   "MDVP:Flo(Hz)":[74.997, 113.819, 111.555, 111.366, 110.655],
                   "MDVP:Jitter(%)":[0.00784, 0.00968, 0.01050, 0.00997, 0.01284],
                   "MDVP:Jitter(Abs) ":[0.00007, 0.00008, 0.00009, 0.00009, 0.00011]})
 
# using quantile() function to
# find the quantiles over the index axis
df.quantile([.1, .25, .5, .75], axis = 0)

"""# Boxplot"""

np.random.seed(10) 
  
data_1 = np.random.normal(100, 10, 200) 
data_2 = np.random.normal(90, 20, 200) 
data_3 = np.random.normal(80, 30, 100) 
data_4 = np.random.normal(70, 40, 100)
data_5 = np.random.normal(60, 50, 100)  
data = [data_1, data_2, data_3, data_4, data_5] 
  
fig = plt.figure(figsize =(10, 7)) 
  
ax = fig.add_axes([0, 0, 1, 1]) 
  
bp = ax.boxplot(data) 
  
plt.show()

"""# Diagrama de dispersión"""

import seaborn
 
 
seaborn.set(style='whitegrid')
fmri = seaborn.load_dataset("fmri")
 
seaborn.scatterplot(x="timepoint",
                    y="signal",
                    data=fmri)

"""# Histograma"""

import random
y=[]

for i in range(10):       
   y.append(random.randint(0, 10))

plt.hist(y, bins = 10, color = "red", rwidth=0.9)
plt.title("Vocal")
plt.xlabel("MDVP:Jitter(%)")
plt.ylabel("MDVP:Jitter(Abs)")
plt.show()

x = range(50)
y = range(50) + np.random.randint(0,30,50)
plt.scatter(x, y)
plt.rcParams.update({'figure.figsize':(10,8), 'figure.dpi':200})
plt.title('Simple Scatter plot')
plt.xlabel('X - MDVP:Jitter(%)')
plt.ylabel('Y - MDVP:Jitter(Abs)')
plt.show()

sns.set_style('dark')
color = ['grey','grey','grey','grey','grey','grey','grey','grey','grey','grey','grey','grey','grey','grey','grey','grey','grey','grey','grey','grey','grey','#D7BDE2']
msno.bar(df,fontsize =14, color = color, sort = 'descending', figsize = (15,10))

plt.text(0.05,1.265,'Parkinson : Null Values', {'size':20, 'weight':'bold'})
plt.text(0.05,1.15,'''Después de realizar el análisis exploratorio, vamos a revisar si existen valores faltantes.''', {'size':12, 'weight':'normal'}, alpha = 0.8)
plt.xticks( rotation = 90, 
                   **{'size':14,'weight':'bold','horizontalalignment': 'center'},alpha = 0.8)

plt.show()

df.columns

"""Se pre-visualiza los datos que se ocuparan dentro de los codigos.

En este apartado se ve los datos de la base de datos obtenida, en este caso la del Parkinson.csv
"""

list_col=['name', 'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)',
       'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
       'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
       'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'status', 'RPDE', 'DFA',
       'spread1', 'spread2', 'D2', 'PPE']

for col in list_col: 
    print('{} :{} ' . format(col.upper(),df[col].unique()))

"""se obtiene tos los datos dentro del de las varibles dadas arriba.

Se hace la correlacion del status y las matrices, para resumir y detectar relaciones en grandes cantidades de datos.
"""

# Compute the correlation matrix
corr = df.corr()

mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(15, 15))

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap='Purples', vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot = True)

"""Se observa en color morado mas intenso la correlación de los datos en mayor cantidad de informacion, los mas tenues o blancos suelen tener menos correlación entre si.

"""

df['status'].value_counts()

"""Data frame se utiiza solo para status ya que sera lo que se utilizara junto con la estructura factores para que me proporcione el total de mis valores dentro de status como una matriz."""

df.drop(columns=['name'], inplace=True)

from sklearn.model_selection import train_test_split
X = df.drop('status', axis = 1)
y = df['status']

X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                   stratify = y,
                                                    test_size = 0.3,
                                                   random_state = 101)

n_random = 7

RF = RandomForestClassifier(n_random)
RF.fit(X_train, y_train)
print('Accuracy of RF classifier on training set: {:.2f}'
     .format(RF.score(X_train, y_train)))
print('Accuracy of RF classifier on test set: {:.2f}'
     .format(RF.score(X_test, y_test)))

RF =  RandomForestClassifier()
RF.fit(X_train,y_train)
predRF = RF.predict(X_test)
cf_matrixRF = confusion_matrix(y_test, predRF)
sns.heatmap(cf_matrixRF/np.sum(cf_matrixRF), annot=True, 
            fmt='.2%', cmap='Reds')
plt.title('Matriz de confusión de  RandomForest', fontweight='bold', fontsize=16)
plt.xlabel('Predicción', fontweight='bold', fontsize=12)
plt.ylabel('Real', fontweight='bold', fontsize=12)

cf_matrixRF

"""La matriz de confusión muestra qué tan bien el árbol separa correctamente las clases utilizando estas métricas:

-Tasa de verdaderos positivos (TPR): la probabilidad de que un caso de evento se pronostique correctamente

-Tasa de falsos positivos (FPR) — la probabilidad de que un caso de no evento se pronostique erróneamente

-Tasa de falsos negativos (FNR) — la probabilidad de que un caso de evento se pronostique erróneamente

-Tasa de verdaderos negativos (TNR): la probabilidad de que un caso de no evento se pronostique correctamente

Su precicion fue de 22.03% en verdaderos positivos y 72.88% en flasos positivos, siendo esta la precision de 94.91.
"""

rfc = RandomForestClassifier(n_estimators=10, random_state=42)
rfc.fit(X_train, y_train)
ax = plt.gca()
rfc_disp = plot_roc_curve(rfc, X_test, y_test, ax=ax, alpha=0.8)
svc_disp.plot(ax=ax, alpha=0.8)
plt.show()

#fprRF, tprRF, thresholds = plot_roc_curve(y_test, predRF)
#plt.figure(figsize=(8,8))
#roc_aucRF = plot.auc(fprRF, tprRF)
#plt.plot(fprRF, tprRF, color='darkorange', label='RF (area = %0.2f)' % roc_aucRF)
#plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
#plt.xlim([0.0, 1.0])
#plt.ylim([0.0, 1.0])
#plt.rcParams['font.size'] = 12
#plt.title('ROC Curve RF', fontweight = 'bold', fontsize=16)
#plt.xlabel('False Positive Rate (1 - Specificity)', fontweight = 'bold', fontsize=14)
#plt.ylabel('True Positive Rate (Sensitivity)', fontweight = 'bold', fontsize=14)
#plt.legend(loc="lower right")
#plt.show()

"""Curva Roc RF se visualizo la separacion ambas variables, siendo estos valores del área bajo la curva ROC normalmente varían de 0.5 a 1. Valores más grandes indican un mejor modelo de clasificación.
La línea de puntos azul indica el caso de asignación aleatoria.

Teniendo un 97% de efectvidad en cuestion a las variables. Siendo este acrdoa la matriz de confucion.

# K-NN

Pasamos a hacer lo mismo con K-NN (k vecinos más cercanos, también conocido como KNN o k-NN), es un clasificador de aprendizaje supervisado no paramétrico, que utiliza la proximidad para hacer clasificaciones o predicciones sobre la agrupación de un punto de datos individual.
"""

n_neighbors = 7

knn = KNeighborsClassifier(n_neighbors)
knn.fit(X_train, y_train)
print('Accuracy of K-NN classifier on training set: {:.2f}'
     .format(knn.score(X_train, y_train)))
print('Accuracy of K-NN classifier on test set: {:.2f}'
     .format(knn.score(X_test, y_test)))

"""El accuracy (exactitud) fue medio-alto, ya que 80% fue de cercanía hacía el valor verdadero.
La desvenatja en su clasificacion, es el algoritmo ya que es perezoso, ocupa más memoria y almacenamiento de datos en comparación con otros clasificadores.
Y esta evaluando los datos entrenamiento solamente. 
"""

KNN =  KNeighborsClassifier()
KNN.fit(X_train,y_train)
predKNN = KNN.predict(X_test)
cf_matrixKNN = confusion_matrix(y_test, predKNN)
sns.heatmap(cf_matrixKNN/np.sum(cf_matrixKNN), annot=True, 
            fmt='.2%', cmap='Reds')
plt.title('Matriz de confusión de  KNeighborsTransformer', fontweight='bold', fontsize=16)
plt.xlabel('Predicción', fontweight='bold', fontsize=12)
plt.ylabel('Real', fontweight='bold', fontsize=12)

cf_matrixKNN

#fprKNN, tprKNN, thresholds = metrics.roc_curve(y_test, predKNN)
#plt.figure(figsize=(8,8))
#roc_aucKNN = metrics.auc(fprKNN, tprKNN)
#plt.plot(fprKNN, tprKNN, color='darkorange', label='RF (area = %0.2f)' % roc_aucRF)
#plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
#plt.xlim([0.0, 1.0])
#plt.ylim([0.0, 1.0])
#plt.rcParams['font.size'] = 12
#plt.title('ROC Curve KNN', fontweight = 'bold', fontsize=16)
#plt.xlabel('False Positive Rate (1 - Specificity)', fontweight = 'bold', fontsize=14)
#plt.ylabel('True Positive Rate (Sensitivity)', fontweight = 'bold', fontsize=14)
#plt.legend(loc="lower right")
#plt.show()

"""en esta curva ROC se demostro la presision de la matriz de confución y como trabajo a pesar de ser un codigo lento o flojo.

teniendo con mejor precision y excatitud, RF.

# TensorFlow

Se usó TensorFlow para el entrenamiento de redes neuronales, para que esta tuvira un mejor desempeño, teniendo un orden y control del codigo en Deep Learning y permite construir y entrenar redes neuronales para detectar patrones y razonamientos usados por los humanos.
"""

#tensorflow
print ("parkinson")
parkinson_df = pd.read_csv("parkinsons.csv")

# Visualización
sns.scatterplot(parkinson_df['status'], parkinson_df['status'])
plt.show()

"""Aqui se demostro el uso de la frecuencia de la voz como ejemplo y parte del entrenamiento. Teniendo 1Mhrz de Frecuencia y al ser dos variables y las dos en base a la voz, solo coloco el punto de incio y final de la emision de la voz."""

#Cargando Set de Datos
print ("selecionado de columnas")
X_train = parkinson_df['name']
y_train = parkinson_df['name']

print ("Creando el modelo")
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=1, input_shape=[1]))

print ("Compilando el modelo")
model.compile(optimizer=tf.keras.optimizers.Adam(1), loss='mean_squared_error')

#Entrenando el modelo
print ("Entrenado")
epochs_hist = model.fit(X_train, y_train, epochs = 0)

"""En estos 4 apartados se entreno el modelo (base), para que redes neuronales pueda enteder como reconocer la voz Humana y poder detectar la anomalia que se encuentre en esta.

# Redes Neuronales

Las Redes Neuronales, pueden ayudar a las computadoras a tomar decisiones inteligentes con asistencia humana limitada. Esto se debe a que pueden aprender y modelar las relaciones entre los datos de entrada y salida que no son lineales y que son complejos.
"""

#redes neuronales 
from scipy import stats

class capa():
  def __init__(self, n_neuronas_capa_anterior, n_neuronas, funcion_act):
    self.funcion_act = funcion_act
    self.b  = np.round(stats.truncnorm.rvs(-1, 1, loc=0, scale=1, size= n_neuronas).reshape(1,n_neuronas),3)
    self.W  = np.round(stats.truncnorm.rvs(-1, 1, loc=0, scale=1, size= n_neuronas * n_neuronas_capa_anterior).reshape(n_neuronas_capa_anterior,n_neuronas),3)

"""En esta parte se lleva el entrenamiento de las Redes Neuronales, ya con lo adquirido de TensorFlow, se empieza a entrenar y empieza a reconocer lo que se le esta pididendo."""

import numpy as np
import math
import matplotlib.pyplot as plt


sigmoid = (
  lambda x:1 / (1 + np.exp(-x)),
  lambda x:x * (1 - x)
  )

rango = np.linspace(-1,1).reshape([50,1])
datos_sigmoide = sigmoid[0](rango)
datos_sigmoide_derivada = sigmoid[1](rango)

#Cremos los graficos
fig, axes = plt.subplots(nrows=1, ncols=2, figsize =(15,5))
axes[0].plot(rango, datos_sigmoide)
axes[1].plot(rango, datos_sigmoide_derivada)
fig.tight_layout()

"""se ecibe un valor x y devuelve un valor entre 0 y 1. Esto hace que sea una función muy interesante, ya que indica la probabilidad de un estado dado, en este caso el 0 y 1.
Se llega a tener -1 para observe el trabajo de las redes en su entrenamiento y ver su función.
"""

def derivada_relu(x):
  x[x<=0] = 0
  x[x>0] = 1
  return x

relu = (
  lambda x: x * (x > 0),
  lambda x:derivada_relu(x)
  )

datos_relu = relu[0](rango)
datos_relu_derivada = relu[1](rango)


# Volvemos a definir rango que ha sido cambiado
rango = np.linspace(-1,1).reshape([50,1])

# Cremos los graficos
plt.cla()
fig, axes = plt.subplots(nrows=1, ncols=2, figsize =(15,5))
axes[0].plot(rango, datos_relu[:,0])
axes[1].plot(rango, datos_relu_derivada[:,0])
plt.show()

"""La función ReLu es muy simple: para valores negativos, la función devuelve cero y funciona en sintonia a las dos valores que se tiene en la base de datos."""

# Numero de neuronas en cada capa. 
# El primer valor es el numero de columnas de la capa de entrada.
neuronas = [2,4,8,1] 

# Funciones de activacion usadas en cada capa. 
funciones_activacion = [relu,relu, sigmoid]

red_neuronal = []

for paso in range(len(neuronas)-1):
  x = capa(neuronas[paso],neuronas[paso+1],funciones_activacion[paso])
  red_neuronal.append(x)

print(red_neuronal)

"""se crear la estructura de nuestra red neuronal y hará de forma iterativa e iremos guardando esta estructura. """

X =  np.round(np.random.randn(20,2),3) # Ejemplo de vector de entrada

z = X @ red_neuronal[0].W

print(z[:10,:], X.shape, z.shape)

"""Para multiplicar los valores de entrada por la matriz de pesos tenemos que hacer una multiplicación matricial."""

z = z + red_neuronal[0].b

print(z[:5,:])

"""Ahora, hay que sumar el parámetro bias (b) al resultado anterior de z."""

a = red_neuronal[0].funcion_act[0](z)
a[:5,:]

"""Ahora, habría que aplicar la función de activación de esa capa."""

output = [X]

for num_capa in range(len(red_neuronal)):
  z = output[-1] @ red_neuronal[num_capa].W + red_neuronal[num_capa].b
  a = red_neuronal[num_capa].funcion_act[0](z)
  output.append(a)

print(output[-1])

"""este es el resultado de la primera capa, que a su vez es la entrada para la segunda capa y así hasta la última. Por tanto, queda bastante claro que todo esto lo podemos definir de forma iterativa dentro de un bucle. """

def mse(Ypredich, Yreal):

  # Calculamos el error
  x = (np.array(Ypredich) - np.array(Yreal)) ** 2
  x = np.mean(x)

  # Calculamos la derivada de la funcion
  y = np.array(Ypredich) - np.array(Yreal)
  return (x,y)

"""Tendríamos la estimación para cada una de las clases de este ejercicio de prueba. Como es la primera ronda, la red no ha entrenado nada, por lo que el resultado es aleatorio. """

from random import shuffle

Y = [0] * 10 + [1] * 10
shuffle(Y)
Y = np.array(Y).reshape(len(Y),1)

mse(output[-1], Y)[0]

"""Las clases (0 o 1) para los valores que nuestra red neuronal ha predicho antes. Asi que calcularemos el error cuadrático medio.

Ahora que ya tenemos el error calculado, tenemos que irlo propagando hacia atrás para ir ajustando los parámetros. Haciendo esto de forma iterativa, nuestra red neuronal irá mejorando sus predicciones, es decir, disminuirá su error. Vamos, que así es como se entrena a una red neuronal.
"""

red_neuronal[-1].b
red_neuronal[-1].W

"""Como desconocemos el valor óptimo de estos parámetros, los hemos inicializado de forma aleatoria. 
Por tanto, en cada ronda estos valores se irán cambiando poco a poco. 
Para ello, lo primero que debemos hacer es transmitir el error hacia atrás.

Osease ir entrenando la red neuronal cada vez hasta su maxima precisión.
"""

# Backprop en la ultima capa
a = output[-1]
x = mse(a,Y)[1] * red_neuronal[-2].funcion_act[1](a)

x

"""ríamos propagando el error generado por la estimación de la red neuronal. Sin embargo, propagar el error por si mismo no hace nada, sino que ahora tenemos que usar ese error para optimizar los valores de los parámetros mediante gradient descent.

gradient descent (Descenso por gradiente) mecanismo de entrenamiento de sistemas de aprendizaje automático como los basados en redes neuronales
"""

red_neuronal[-1].b = red_neuronal[-1].b - x.mean() * 0.01
red_neuronal[-1].W = red_neuronal[-1].W - (output[-1].T @ x) * 0.01

red_neuronal[-1].b
red_neuronal[-1].W

# Definimos el learning rate
lr = 0.05

# Creamos el indice inverso para ir de derecha a izquierda
back = list(range(len(output)-1))
back.reverse()

# Creamos el vector delta donde meteremos los errores en cada capa
delta = []

for capa in back:
  # Backprop #

  # Guardamos los resultados de la ultima capa antes de usar backprop para poder usarlas en gradient descent
  a = output[capa+1][1]

  # Backprop en la ultima capa 
  if capa == back[0]:
    x = mse(a,Y)[1] * red_neuronal[capa].funcion_act[1](a)
    delta.append(x)

  # Backprop en el resto de capas 
  else:
    x = delta[-1] @ W_temp * red_neuronal[capa].funcion_act[1](a)
    delta.append(x)

  # Guardamos los valores de W para poder usarlos en la iteracion siguiente
  W_temp = red_neuronal[capa].W.transpose()

  # Gradient Descent #

  # Ajustamos los valores de los parametros de la capa
  red_neuronal[capa].b = red_neuronal[capa].b - delta[-1].mean() * lr
  red_neuronal[capa].W = red_neuronal[capa].W - (output[capa].T @ delta[-1]) * lr


print('MSE: ' + str(mse(output[-1],Y)[0]) )
print('Estimacion: ' + str(output[-1]) )

"""Tendríamos aplicado el backpropagation y gradient descent.

backpropagation es el método que emplea un ciclo de propagación–adaptación de dos fases, en resumen permite que la información del costo fluya hacia atrás a través de la red para calcular el gradiente.1​ Una vez que se ha aplicado un patrón a la entrada de la red como estímulo, este se propaga desde la primera capa a través de las capas siguientes de la red, hasta generar una salida.
"""

import random

def circulo(num_datos = 196,R = 1, minimo = 0,maximo= 1):
  pi = math.pi
  r = R * np.sqrt(stats.truncnorm.rvs(minimo, maximo, size= num_datos)) * 10
  theta = stats.truncnorm.rvs(minimo, maximo, size= num_datos) * 2 * pi *10

  x = np.cos(theta) * r
  y = np.sin(theta) * r

  y = y.reshape((num_datos,1))
  x = x.reshape((num_datos,1))

  #Vamos a reducir el numero de elementos para que no cause un Overflow
  x = np.round(x,3)
  y = np.round(y,3)

  df = np.column_stack([x,y])
  return(df)

"""Se crearemos dos sets de datos aleatorios, cada uno de 196 puntos y con radios diferentes. La idea de hacer que los datos se creen de forma aleatoria es que puedan solaparse, de tal manera que a la red neuronal le cueste un poco y el resultado no sea perfecto."""

datos_1 = circulo(num_datos = 196, R = 2)
datos_2 = circulo(num_datos = 196, R = 0.5)
X = np.concatenate([datos_1,datos_2])
X = np.round(X,3)
#print(X.shape) 
Y = [0] * 196 + [1] * 196
Y = np.array(Y).reshape(len(Y),1)
#print(Y.shape)

"""Con esto ya tendríamos nuestros datos de entrada (X) y sus correspondientes etiquetas (Y). Teniendo esto en cuenta, visualicemos cómo es el problema que debe resolver nuestra red neuronal.
Donde los positivos (1) son el total de 148 y el Total de (0) son 48, su suma nos da 196 datos los cuales se ocuparan.
"""

plt.cla()
plt.scatter(X[0:196,1],X[0:196,0], c = "r")
plt.scatter(X[196:392,0],X[196:392,0], c = "b")
plt.show()

"""Se marca con azul los caso positivos,
Y de rojo falsos positivos

"""

def entrenamiento(X,Y, red_neuronal, lr = 0.01):

  # Output guardara el resultado de cada capa
  # En la capa 1, el resultado es el valor de entrada
  output = [X]

  for num_capa in range(len(red_neuronal)):
    z = output[-1] @ red_neuronal[num_capa].W + red_neuronal[num_capa].b

    a = red_neuronal[num_capa].funcion_act[0](z)

    # Incluimos el resultado de la capa a output
    output.append(a)

  # Backpropagation

  back = list(range(len(output)-1))
  back.reverse()

  # Guardaremos el error de la capa en delta  
  delta = []

  for capa in back:
    # Backprop #delta

    a = output[capa+1]

    if capa == back[0]:
      x = mse(a,Y)[1] * red_neuronal[capa].funcion_act[1](a)
      delta.append(x)

    else:
      x = delta[-1] @ W_temp * red_neuronal[capa].funcion_act[1](a)
      delta.append(x)

    W_temp = red_neuronal[capa].W.transpose()

    # Gradient Descent #
    red_neuronal[capa].b = red_neuronal[capa].b - np.mean(delta[-1], axis = 0, keepdims = True) * lr
    red_neuronal[capa].W = red_neuronal[capa].W - output[capa].transpose() @ delta[-1] * lr

  return output[-1]

"""Entrenamientos para obtener mayor precision de la red neuronal."""

class capa():
  def __init__(self, n_neuronas_capa_anterior, n_neuronas, funcion_act):
    self.funcion_act = funcion_act
    self.b  = np.round(stats.truncnorm.rvs(-1, 1, loc=0, scale=1, size= n_neuronas).reshape(1,n_neuronas),3)
    self.W  = np.round(stats.truncnorm.rvs(-1, 1, loc=0, scale=1, size= n_neuronas * n_neuronas_capa_anterior).reshape(n_neuronas_capa_anterior,n_neuronas),3)

neuronas = [2,4,8,1] 
funciones_activacion = [relu,relu, sigmoid]
red_neuronal = []

for paso in list(range(len(neuronas)-1)):
  x = capa(neuronas[paso],neuronas[paso+1],funciones_activacion[paso])
  red_neuronal.append(x)

error = []
predicciones = []

for epoch in range(0,10):
  ronda = entrenamiento(X = X ,Y = Y ,red_neuronal = red_neuronal, lr = 0.0001)
  predicciones.append(ronda)
  temp = mse(np.round(predicciones[-1]),Y)[0]
  error.append(temp)

epoch = list(range(0,10))
plt.plot(epoch, error)

"""Y aqui se obtiene el codigo ya entrenado de redes neuronales, donde muestra el el rango de error en datos aleatorios, ya que muestra solo un dato aleatorio de 0 ó 1.

# Visualizacion de datos especificos
"""

# Commented out IPython magic to ensure Python compatibility.

# Imports necesarios
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
# %matplotlib inline
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

"""Se manda a llamar la base de datos necesaria para regresion lineal"""

#cargamos los datos de entrada
data = pd.read_csv("parkinsons.csv")
#veamos cuantas dimensiones y registros contiene
data.shape

"""Se carga la base datos con la matriz de confucion"""

#son 161 registros con 8 columnas. Veamos los primeros registros
data.head()

"""Se visualiza los primeros registros."""

# Ahora veamos algunas estadísticas de nuestros datos
data.describe()

"""Se visualizan los primeros datos

"""

# Visualizamos rápidamente las caraterísticas de entrada
data.drop(['NHR','HNR', 'status'],1).hist()
plt.show()

"""se observa las graficas obtenidas de las variables

# Capitlo VI

# Regresion Logistica

Esta resulta útil para los casos en los que se desea predecir la presencia o ausencia de una característica o resultado según los valores de un conjunto de predictores. Es similar a un modelo de regresión lineal pero está adaptado para modelos en los que la variable dependiente es dicotómica.
"""

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
# Generamos un dataset de dos clases
X, y = make_classification(n_samples=1000, n_classes=2, random_state=1)
# Dividimos en training y test
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=2)
#Generamos un clasificador sin entrenar , que asignará 0 a todo
ns_probs = [0 for _ in range(len(testy))]
# Entrenamos nuestro modelo de reg log
model = LogisticRegression(solver='lbfgs')
model.fit(trainX, trainy)
# Predecimos las probabilidades
lr_probs = model.predict_proba(testX)
#Nos quedamos con las probabilidades de la clase positiva (la probabilidad de 1)
lr_probs = lr_probs[:, 1]
# Calculamos el AUC
ns_auc = roc_auc_score(testy, ns_probs)
lr_auc = roc_auc_score(testy, lr_probs)
# Imprimimos en pantalla
print('Sin entrenar: ROC AUC=%.3f' % (ns_auc))
print('Regresión Logística: ROC AUC=%.3f' % (lr_auc))
# Calculamos las curvas ROC
ns_fpr, ns_tpr, _ = roc_curve(testy, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(testy, lr_probs)
# Pintamos las curvas ROC
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='Sin entrenar')
pyplot.plot(lr_fpr, lr_tpr, marker='.', label='Regresión Logística')
# Etiquetas de los ejes
pyplot.xlabel('Tasa de Falsos Positivos')
pyplot.ylabel('Tasa de Verdaderos Positivos')
pyplot.legend()

"""en este modelo se implemento la Curva Roc, donde se observa la tasa de verdaderos positivos y Falsos positivos, donde la tasa de flaso positivos, donde la linea roja indica el caso de asignación aleatoria sobre los falsos positivos, dando a entender la sensibilidad."""

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from matplotlib import pyplot
#Generamos dataset
X, y = make_classification(n_samples=1000, n_classes=2, random_state=1)
#Dividimos en training y test
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=2)
#Entrenamos
model = LogisticRegression(solver='lbfgs')
model.fit(trainX, trainy)
# predecimos probabilidades
lr_probs = model.predict_proba(testX)
# Nos quedamos unicamente con las predicciones positicas
lr_probs = lr_probs[:, 1]
# Sacamos los valores
yhat = model.predict(testX)
lr_precision, lr_recall, _ = precision_recall_curve(testy, lr_probs)
lr_f1, lr_auc = f1_score(testy, yhat), auc(lr_recall, lr_precision)
# Resumimos s
print('Regresión Logística: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
# plot the precision-recall curves
no_skill = len(testy[testy==1]) / len(testy)
pyplot.plot([0, 1], [no_skill, no_skill], linestyle='--', label='Sin entrenar')
pyplot.plot(lr_recall, lr_precision, marker='.', label='Regresión Logística')
#Etiquetas de ejes
pyplot.xlabel('Sensibilidad')
pyplot.ylabel('Precisión')
pyplot.legend()
pyplot.show()

"""En este modelo se observo  accuracy o la exacitud de 89%. Y Exactitud es  la cantidad de predicciones positivas que fueron correctas.

La sensibilidad da los valores que nos indican la capacidad de nuestro estimador para discriminar los casos positivos, de los negativos.

La precision nos indica la dispersión del conjunto de valores obtenidos a partir de mediciones repetidas de una magnitud. Cuanto menor es la dispersión mayor la precisión. 
"""

#Importamos 
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
# Generamos un dataset de dos clases (desbalanceadas en un 99:1)
X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.99,0.01], random_state=1)
# Dividimos en training y test
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=2)
#Generamos un clasificador sin entrenar , que asignará 0 a todo
ns_probs = [0 for _ in range(len(testy))]
# Entrenamos nuestro modelo de reg log
model = LogisticRegression(solver='lbfgs')
model.fit(trainX, trainy)
# Predecimos las probabilidades
lr_probs = model.predict_proba(testX)
#Nos quedamos con las probabilidades de la clase positiva (la probabilidad de 1)
lr_probs = lr_probs[:, 1]
# Calculamos el AUC
ns_auc = roc_auc_score(testy, ns_probs)
lr_auc = roc_auc_score(testy, lr_probs)
# Imprimimos en pantalla
print('Sin entrenar: ROC AUC=%.3f' % (ns_auc))
print('Regresión Logística: ROC AUC=%.3f' % (lr_auc))
# Calculamos las curvas ROC
ns_fpr, ns_tpr, _ = roc_curve(testy, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(testy, lr_probs)
# Pintamos las curvas ROC
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='Sin entrenar')
pyplot.plot(lr_fpr, lr_tpr, marker='.', label='Regresión Logística')
# Etiquetas de los ejes
pyplot.xlabel('Tasa de Falsos Positivos')
pyplot.ylabel('Tasa de Verdaderos Positivos')
pyplot.legend()
pyplot.show()

yhat = model.predict(testX)
lr_precision, lr_recall, _ = precision_recall_curve(testy, lr_probs)
lr_f1, lr_auc = f1_score(testy, yhat), auc(lr_recall, lr_precision)
# Resumimos s
print('Regresión Logística: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
# Pintamos la curva de precision-sensibilidad curves
no_skill = len(testy[testy==1]) / len(testy)
pyplot.plot([0, 1], [no_skill, no_skill], linestyle='--', label='Sin entrenar')
pyplot.plot(lr_recall, lr_precision, marker='.', label='Regresión Logística')
#Etiquetas de ejes
pyplot.xlabel('Sensibilidad')
pyplot.ylabel('Precisión')
pyplot.legend()

from sklearn.model_selection import train_test_split
#Separo los datos de "train" en entrenamiento y prueba para probar los algoritmos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#Defino el algoritmo a utilizar
from sklearn.svm import SVC
algoritmo = SVC(kernel = 'linear')

"""# Capitulo VII

# Maquinas de soporte Vectorial.

se da en la clasificación binaria, es decir para separar un set de datos en dos categorías o clases diferente, su clasificación mediante vecinos más cercanos, y el modelo de regresión lineal, Pronóstico de datos numéricos, métodos de regresión. Esto hace que combinación es extremadamente poderosa, lo que permite a las SVM modelar relaciones altamente complejas.
"""

#Entreno el modelo
algoritmo.fit(X_train, y_train)

#Realizo una predicción
y_pred = algoritmo.predict(X_test)

#Verifico la matriz de Confusión
from sklearn.metrics import confusion_matrix
matriz = confusion_matrix(y_test, y_pred)
print('Matriz de Confusión:')
print(matriz)

#Calculo la precisión del modelo
from sklearn.metrics import precision_score
precision = precision_score(y_test, y_pred)
print('Precisión del modelo:')
print(precision)

"""
Máquinas Vectores de Soporte Clasificación
"""
########## LIBRERÍAS A UTILIZAR ##########
#Se importan la librerias a utilizar
from sklearn import datasets
########## PREPARAR LA DATA ##########
#Importamos los datos de la misma librería de scikit-learn
dataset = datasets.load_breast_cancer()
print(dataset)
########## ENTENDIMIENTO DE LA DATA ##########
#Verifico la información contenida en el dataset
print('Información en el dataset:')
print(dataset.keys())
print()
#Verifico las características del dataset
print('Características del dataset:')
print(dataset.DESCR)
#Seleccionamos todas las columnas
X = dataset.data
#Defino los datos correspondientes a las etiquetas
y = dataset.target
########## IMPLEMENTACIÓN DE MAQUINAS VECTORES DE SOPORTE ##########
from sklearn.model_selection import train_test_split
#Separo los datos de "train" en entrenamiento y prueba para probar los algoritmos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#Defino el algoritmo a utilizar
from sklearn.svm import SVC
algoritmo = SVC(kernel = 'linear')
#Entreno el modelo
algoritmo.fit(X_train, y_train)
#Realizo una predicción
y_pred = algoritmo.predict(X_test)
#Verifico la matriz de Confusión
from sklearn.metrics import confusion_matrix
matriz = confusion_matrix(y_test, y_pred)
print('Matriz de Confusión:')
print(matriz)
#Calculo la precisión del modelo
from sklearn.metrics import precision_score
precision = precision_score(y_test, y_pred)
print('Precisión del modelo:')
print(precision)

"""Me da una precision de 94-95%, la razon es que al tener dos variables 1 y 0 llega a desempeñarse mejor al solo tener una linealidad.

# *Capitulo VIII*

# Encontrando Patrones
"""

# print 0 to 1
for i in range(2):
    print(i, end=' ')

for i in range(1):
    for j in range(2):
        print(j+1, end=' ')
    print() # new line

n = 1
for i in range(n+1):
    for j in range(1, i+2):
        print(j, end=' ')
    print()

n = 2
for i in range(n+1):
    for j in range(1, i+2):
        print(j, end=' ')
    print()

n = 2
for i in range(n+1):
    for j in range(1, i+2):
        print(i, end=' ')
    print()

"""Se observa que el codigo no es funcional, el echo es que no es un Big data, o un grana campo de almacenamiento de datos, donde se observa que tenemos solo dos variables utilizada de nuestro status.

#*Capitulo IX*

# algoritmo k-means
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(7)

x1 = np.random.standard_normal((196,2))*0.6+np.ones((196,2))
x2 = np.random.standard_normal((196,2))*0.5-np.ones((196,2))
#x3 = np.random.standard_normal((100,2))*0.4-2*np.ones((100,2))+5
X = np.concatenate((x1,x2),axis=0)

plt.plot(X[:,0],X[:,1],'k.')
plt.show()

"""Implementar el método k-means de forma que clasifique los 2 datos como datos de dos dimensiones."""

from sklearn.cluster import KMeans

n = 2
k_means = KMeans(n_clusters=n)
k_means.fit(X)

centroides = k_means.cluster_centers_
etiquetas = k_means.labels_

plt.plot(X[etiquetas==0,0],X[etiquetas==0,1],'r.', label='cluster 1')
plt.plot(X[etiquetas==1,0],X[etiquetas==1,1],'b.', label='cluster 2')

plt.plot(centroides[:,0],centroides[:,1],'mo',markersize=8, label='centroides')

plt.legend(loc='best')
plt.show()

"""# *Capitulo X*

# Evaluación de Modelos Predictivos

# *Capitulo XI*

# Modelo de mejora Actuación

# *Capitulo XII*

# Máquina especializada Temas de aprendizaje
"""

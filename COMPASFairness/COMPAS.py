import numpy as np # linear algebra
import pandas as pd# data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import warnings
from sklearn.model_selection import train_test_split
warnings.simplefilter(action='ignore', category=FutureWarning)
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler


#Primero leer los datos

data = pd.read_csv("/Users/pilarlahoz/Documents/TFG/propublica_data_for_fairml.csv")

#Ahora queremos descubrir los datos
print(data.shape)
print(data.columns)
data.head()
data.info()
data.describe()

#Vamos a ver como se distribuyen los datos
data.hist(bins=50, figsize=(20,15))
plt.show()

#Vamos a preprocesar los datos para que sean mejores para el entrenamiento
#### Buscar filas nan y reemplazarlas por la media
null_columns = data.columns[data.isnull().any()] #filas con NaN
data[null_columns].isnull().sum()
print(data[data.isnull().any(axis=1)][null_columns].head())

data = data.fillna(data.mean()) #rellenarlo con la media de todo el grupo

# Visualizar y ver su correlación
# Dividimos ya el test_set y el train_set(que se dividira x_train y X_valid)
aux_set, test_set = train_test_split(data, test_size=0.2, random_state=42)

corr_matrix = test_set.corr()
corr_matrix["Two_yr_Recidivism"].sort_values(ascending=False)


corr = test_set.set_index('Two_yr_Recidivism').corr()
sm.graphics.plot_corr(corr, xnames=list(corr.columns))
plt.show() # Al ser un dataset binario es complicado visualizar los datos graficamente

# Prepara los datos para los algoritmos de machine learning
TARGET_COL = "Two_yr_Recidivism"

# eliminamos la columna de reincidencia de dos años de los datos para añadirlo a la y
x_aux = aux_set.drop([TARGET_COL],axis=1)
y_aux = aux_set[TARGET_COL]

# Create train and validation set
train_x, valid_x, train_y, valid_y = train_test_split(x_aux, y_aux, test_size=0.25, shuffle=True, stratify=y_aux, random_state=42)

print("El tamaño de la muestra de entreno es",train_x.size)
print("El tamaño de la muestra de validación es", valid_x.size)

#Escalar los datos
scaler = MinMaxScaler()
scaler.fit(train_x)

#Escalamos train_x
processtrain_x = scaler.transform(train_x)

#Escalamos valid_x
processvalid_x = scaler.transform(valid_x)

#Buscamos con qué columna calculamos los distintos algoritmos de Fairness

plt.style.use("bmh")            #Declaración del estilo
x_values = valid_x['African_American'].unique()
y_values = valid_x['African_American'].value_counts().tolist()
plt.bar(x_values, y_values)          #El gráfico
plt.title('African American')      #El título
ax = plt.subplot()                   #Axis
ax.set_xticks(x_values)             #Eje x
ax.set_xticklabels(x_values)        #Etiquetas del eje x
ax.set_xlabel('African American')  #Nombre del eje x
ax.set_ylabel('Numero de personas')  #Nombre del eje y
plt.show()
plt.close('all')

#Elijo la categoria de African_American

#De todas las soluciones quiero que sea las que tienen un 1 en african_american
#African American es la columna 4 en porcessdata

indice_aa = processvalid_x[:,4]==1
indice_noaa = processvalid_x[:,4]==0

grupo_aa = processvalid_x[indice_aa]

grupo_noaa = processvalid_x[indice_noaa]

print ("grupo afroamericanos:", len(grupo_aa), "grupo no afroamericanos:", len(grupo_noaa))

#creamos valid_y
valid_yaux1= valid_y[indice_aa]
valid_yaux2= valid_y[indice_noaa]
valid_yaux = np.concatenate((valid_yaux1, valid_yaux2))

print("afroamericanos reincidente:", np.size(valid_yaux1[valid_yaux1 ==1]), "no afroamericanos reincidente", np.size(valid_yaux2[valid_yaux2 ==1]))





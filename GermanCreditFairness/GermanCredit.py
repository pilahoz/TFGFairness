import numpy as np # linear algebra
import pandas as pd# data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
warnings.simplefilter(action='ignore', category=FutureWarning)
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler

#Primero leer los datos

data = pd.read_csv("/Users/pilarlahoz/Documents/TFG/german_credit_data.csv")

#Ahora queremos descubrir los datos
print(data.shape)
print(data.columns)
data.head()
data.info()
data.describe()

del data["Unnamed: 0"] #eliminamos esa columna ya que no sirve de nada

#Vamos a ver como se distribuyen los datos
data.hist()
plt.show()

#Vamos a preprocesar los datos para que sean mejores para el entrenamiento
#### Buscar filas nan y reemplazarlas por la media
null_columns=data.columns[data.isnull().any()] #filas con NaN
data[null_columns].isnull().sum()
print(data[data.isnull().any(axis=1)][null_columns].head())

data = data.fillna(data.mean()) #rellenarlo con la media de todo el grupo

#Aquí vamos a hacer columnas nuevas para que todas sean numericas

interval = (0, 33, 120)

cats = ['Young', 'Adult']
data["Age_cat"] = pd.cut(data.Age, interval, labels=cats)

data_good =data[data["Risk"] == 'good']
data_bad = data[data["Risk"] == 'bad']

data['Saving accounts'] = data['Saving accounts'].fillna('no_inf')
data['Checking account'] = data['Checking account'].fillna('no_inf')

#Purpose to Dummies Variable
data = data.merge(pd.get_dummies(data.Purpose, drop_first=True, prefix='Purpose'), left_index=True, right_index=True)
#Feature in dummies
data = data.merge(pd.get_dummies(data.Sex, drop_first=True, prefix='Sex'), left_index=True, right_index=True)
# Housing get dummies
data = data.merge(pd.get_dummies(data.Housing, drop_first=True, prefix='Housing'), left_index=True, right_index=True)
# Housing get Saving Accounts
data = data.merge(pd.get_dummies(data["Saving accounts"], drop_first=True, prefix='Savings'), left_index=True, right_index=True)
# Housing get Risk
data = data.merge(pd.get_dummies(data.Risk, prefix='Risk'), left_index=True, right_index=True)
# Housing get Checking Account
data = data.merge(pd.get_dummies(data["Checking account"], drop_first=True, prefix='Check'), left_index=True, right_index=True)
# Housing get Age categorical
data = data.merge(pd.get_dummies(data["Age_cat"], drop_first=True, prefix='Age_cat'), left_index=True, right_index=True)

print(data.columns) #vemos cuantas columnas tenemos y sobran

#Eliminamos las columnas que sobran
del data["Saving accounts"]
del data["Checking account"]
del data["Purpose"]
del data["Sex"]
del data["Housing"]
del data["Age_cat"]
del data["Risk"]
del data["Risk_bad"]

# Visualizar y ver su correlación
# Dividimos ya el test_set y el train_set(que se dividira x_train y X_valid)
aux_set, test_set = train_test_split(data, test_size=0.2, random_state=42)

corr_matrix = test_set.corr()
corr_matrix["Risk_good"].sort_values(ascending=False)


corr = test_set.set_index('Risk_good').corr()
sm.graphics.plot_corr(corr, xnames=list(corr.columns))
plt.show() # Al ser un dataset binario es complicado visualizar los datos graficamente

# Prepara los datos para los algoritmos de machine learning
TARGET_COL = "Risk_good"

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

print(valid_x.columns)

#Buscamos con qué columna calculamos los distintos algoritmos de Fairness

column_data = valid_x.iloc[:, valid_x.columns.get_loc('Sex_male')]

#vamos a ver el Sex_male
plt.style.use("bmh")            #Declaración del estilo
x_values = valid_x['Sex_male'].unique()
y_values = valid_x['Sex_male'].value_counts().tolist()
plt.bar(column_data.value_counts().index, column_data.value_counts().values)  #El grafico
plt.title('Sex_male')      #El título
ax = plt.subplot()                   #Axis
ax.set_xticks(x_values)             #Eje x
ax.set_xticklabels(x_values)        #Etiquetas del eje x
ax.set_xlabel('Sex_male')  #Nombre del eje x
ax.set_ylabel('Numero de personas')  #Nombre del eje y
plt.show()
plt.close('all')


#De todas las soluciones quiero que sea las que tienen un 1 en Sex_male
#Sex_male es la columna 12 (como empieza en 0, 11)

indice_h = processvalid_x[:,11]==1
indice_m = processvalid_x[:,11]==0

grupo_h = processvalid_x[indice_h]

grupo_m = processvalid_x[indice_m]
#Elijo la categoria de Sex_male

#creamos valid_y
valid_yaux1= valid_y[indice_h]
valid_yaux2= valid_y[indice_m]
valid_yaux = np.concatenate((valid_yaux1, valid_yaux2))

rows1, columns1 = grupo_h.shape
rows2, columns1 = grupo_m.shape

print("numero de hombres:", rows1 ,"con credito:", np.size(valid_yaux1[valid_yaux1 ==1]))
print("numero de mujeres:",rows2,"con credito:", np.size(valid_yaux2[valid_yaux2==1]))






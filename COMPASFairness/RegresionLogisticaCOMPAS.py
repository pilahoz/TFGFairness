from matplotlib import pyplot as plt
from numpy import sqrt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn import metrics

import numpy as np # linear algebra
import pandas as pd
import COMPAS
import Fairness
import math


#Vamos crear el algoritmo de machine learning -> Regresion Logisticas
log_reg = LogisticRegression()
log_reg.get_params()


#Hacemos un gridsearh para encontrar los mejores hiperparámetros
parameters={"C":np.logspace(-3,3,7), "penalty":["l2"]}
gslog_reg=GridSearchCV(log_reg,parameters,cv=5) #cv = cross validation
gslog_reg.fit(COMPAS.processtrain_x, COMPAS.train_y) #Entrenamos el algoritmo de machine learning
print("tuned hpyerparameters :(best parameters) ",gslog_reg.best_params_)

pd.DataFrame(gslog_reg.cv_results_).head()


#Calculamos el score

score = gslog_reg.score(COMPAS.processvalid_x, COMPAS.valid_y) #Devuelve la precisión media en los datos de prueba y las etiquetas dadas
print("The score for this model is ", score)

#Matriz de confusion
pred_y = gslog_reg.predict(COMPAS.processvalid_x) #Prediccion

print('The prediction is:', pred_y)
print('Accuracy: %.3f' % accuracy_score(COMPAS.valid_y, pred_y))

cm = confusion_matrix(COMPAS.valid_y, pred_y)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=gslog_reg.classes_)
disp.plot()
plt.show()

# Calculamos la curva ROC

y_pred_proba = gslog_reg.predict_proba(COMPAS.processvalid_x)[:, 1] # Obtenemos las probabilidades de predicción del modelo para el conjunto de prueba

fpr, tpr, umbrales = roc_curve(COMPAS.valid_y, y_pred_proba) #calculamos los valores para hacer la curva

roc_auc = auc(fpr, tpr) # Calculamos el área bajo la curva ROC
print("---------------------------------auc:", roc_auc)

# Graficamos la curva ROC
plt.plot(fpr, tpr, color='darkorange', lw=2, label='Curva ROC (área = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de falsos positivos (FPR)')
plt.ylabel('Tasa de verdaderos positivos (TPR)')
plt.title('Curva ROC para modelo de regresión logística')
plt.legend(loc="lower right")
plt.show()


# Calculamos la distancia entre cada punto de la curva ROC y la recta y = x
distances = [math.sqrt((fpr[i])**2 + (1 - tpr[i])**2) for i in range(len(fpr))]

# Calculamos la distancia mínima
min_distance = min(distances)

# Imprimimos el valor mínimo de la distancia
print("La distancia mínima entre los puntos de la curva ROC y el punto (0,1) es:", min_distance)

# Obtenemos el índice del valor mínimo en la lista de distancias
min_index = distances.index(min(distances))

# Obtenemos el punto correspondiente en las listas de FPR y TPR
min_fpr = fpr[min_index]
min_tpr = tpr[min_index]

# Imprimimos el punto con la distancia mínima
print("El punto de la curva ROC con la distancia mínima al punto (0,1) es:", min_fpr, min_tpr)

#El umbral al que pertenece
min_umbral = umbrales[min_index]

print("El mejor umbral para este modelo de machine learning es:", min_umbral)


#--------------------------------------------------------------------------------------------#

#Fairness


#Sacamos las filas para ver la paridad
rows1, columns1 = COMPAS.grupo_aa.shape
rows2, columns1 = COMPAS.grupo_noaa.shape
print("filas grupo African_American:", rows1)
print("filas grupo no African_American:", rows2)


#CALCULAMOS PROBABILIDADES
p1_hat = gslog_reg.predict_proba(COMPAS.grupo_aa) #probabilidad del grupo_aa
p2_hat = gslog_reg.predict_proba(COMPAS.grupo_noaa) #probabilidad del grupo_noaa

#-------PARIDAD-------
umbral_aap, umbral_noaap =Fairness.umbralparidad(COMPAS.grupo_aa, COMPAS.grupo_noaa, p1_hat, p2_hat, min_umbral)
print("umbra aa:",umbral_aap, "umbral noaa:", umbral_noaap)

#columna de prob que si reincidan
ind1 = p1_hat[:,1]>umbral_aap
ind2 = p2_hat[:,1]>umbral_noaap

#concatenar las indices
indices = np.concatenate((ind1,ind2))

#grupos de los si reincidentes
grupo1 = COMPAS.grupo_aa[ind1, :]
grupo2 = COMPAS.grupo_noaa[ind2, :]

#calculo de las filas
rows1, columns1 = grupo1.shape
rows2, columns2 = grupo2.shape
print("filas grupo African_American que superan el umbral:", rows1)
print("filas grupo no African_American que superan el umbral:", rows2)

p_y = Fairness.predictumbral(indices, COMPAS.valid_yaux) #con ese umbral calculamos la prediccion
print('Accuracy: %.3f' % accuracy_score(p_y, COMPAS.valid_yaux))

#vemos su matriz de confusion
cm = confusion_matrix(p_y, COMPAS.valid_yaux)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=gslog_reg.classes_)
disp.plot()
plt.show()

#-------IGUALDAD DE PROBABILIDADES-------
umbral_aaip, umbral_noaaip = Fairness.umbralprobabilidades(COMPAS.grupo_aa, COMPAS.grupo_noaa, p1_hat, p2_hat, gslog_reg, COMPAS.valid_yaux1, COMPAS.valid_yaux2, COMPAS.valid_yaux, min_umbral)
print("umbra aa:",umbral_aaip, "umbral noaa:", umbral_noaaip)

ind1ip = p1_hat[:,1]>umbral_aaip #columna de prob que si reincidan
ind2ip = p2_hat[:,1]>umbral_noaaip

indicesig = np.concatenate((ind1ip,ind2ip))

p_yip = Fairness.predictumbral(indicesig, COMPAS.valid_yaux) #con ese umbral calculamos la prediccion
print('Accuracy: %.3f' % accuracy_score(p_yip, COMPAS.valid_yaux))

#vemos su matriz de confusion
cm = confusion_matrix(p_yip, COMPAS.valid_yaux)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=gslog_reg.classes_)
disp.plot()
plt.show()

#-------IGUALDAD DE OPORTUNIDADES-------
umbral_aaio, umbral_noaaio = Fairness.umbraloportunidades(COMPAS.grupo_aa, COMPAS.grupo_noaa, p1_hat, p2_hat, gslog_reg, COMPAS.valid_yaux1, COMPAS.valid_yaux2, COMPAS.valid_yaux, min_umbral)
print("umbra aa:",umbral_aaio, "umbral noaa:", umbral_noaaio)

ind1io = p1_hat[:,1]>umbral_aaio #columna de prob que si reincidan
ind2io = p2_hat[:,1]>umbral_noaaio

indicesio = np.concatenate((ind1io,ind2io))

p_yio = Fairness.predictumbral(indicesio, COMPAS.valid_yaux) #con ese umbral calculamos la prediccion
print('Accuracy: %.3f' % accuracy_score(p_yio, COMPAS.valid_yaux))

#vemos su matriz de confusion
cm = confusion_matrix(p_yio, COMPAS.valid_yaux)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=gslog_reg.classes_)
disp.plot()
plt.show()

#-------IMPACTO DESIGUAL-------
umbral_aaid, umbral_noaaid = Fairness.umbralimpactodesigual(COMPAS.grupo_aa, COMPAS.grupo_noaa, p1_hat, p2_hat, gslog_reg, COMPAS.valid_yaux1, COMPAS.valid_yaux2, COMPAS.valid_yaux, min_umbral)
print("umbra aa:",umbral_aaid, "umbral noaa:", umbral_noaaid)

ind1id = p1_hat[:,1]>umbral_aaid #columna de prob que si reincidan
ind2id = p2_hat[:,1]>umbral_noaaid

indicesid = np.concatenate((ind1id,ind2id))

p_yid = Fairness.predictumbral(indicesid, COMPAS.valid_yaux) #con ese umbral calculamos la prediccion
print('Accuracy: %.3f' % accuracy_score(p_yid, COMPAS.valid_yaux))

#vemos su matriz de confusion
cm = confusion_matrix(p_yid, COMPAS.valid_yaux)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=gslog_reg.classes_)
disp.plot()
plt.show()






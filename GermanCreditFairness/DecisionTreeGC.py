import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import GermanCredit
import Fairness
import math

#Vamos crear el algoritmo de machine learning -> Decision Tree
dec_tree = DecisionTreeClassifier()
dec_tree.get_params()

#Hacemos un gridsearh para encontrar los mejores hiperparámetros
param_grid = { 'criterion':['gini','entropy'],'max_depth': np.arange(3, 15)}

gsdec_tree= GridSearchCV(dec_tree, param_grid, cv=5)
gsdec_tree.fit(GermanCredit.processtrain_x, GermanCredit.train_y) #Entrenamos el algoritmo de machine learning
print("tuned hpyerparameters :(best parameters) ",gsdec_tree.best_params_)

pd.DataFrame(gsdec_tree.cv_results_).head()
#Calculamos el score

#Devuelve la precisión media en los datos de prueba y las etiquetas dadas
score = gsdec_tree.score(GermanCredit.processvalid_x, GermanCredit.valid_y)
print ("The score for this model is ", score)

#Matriz de confusion

pred_y = gsdec_tree.predict(GermanCredit.processvalid_x) #Prediccion

print('The prediction is:', pred_y)
print('Accuracy: %.3f' % accuracy_score(GermanCredit.valid_y, pred_y))

cm = confusion_matrix(GermanCredit.valid_y, pred_y)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=gsdec_tree.classes_)
disp.plot()
plt.show()

y_pred_proba = gsdec_tree.predict_proba(GermanCredit.processvalid_x)[:, 1] # Obtenemos las probabilidades de predicción del modelo para el conjunto de prueba

fpr, tpr, umbrales = roc_curve(GermanCredit.valid_y, y_pred_proba) #calculamos los valores para hacer la curva
print(fpr,tpr, umbrales)

roc_auc = auc(fpr, tpr) # Calculamos el área bajo la curva ROC
print("auc:", roc_auc)

# Graficamos la curva ROC
plt.plot(fpr, tpr, color='darkorange', lw=2, label='Curva ROC (área = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de falsos positivos (FPR)')
plt.ylabel('Tasa de verdaderos positivos (TPR)')
plt.title('Curva ROC para modelo de Decision Tree')
plt.legend(loc="lower right")
plt.show()


# Calculamos la distancia entre cada punto de la curva ROC y la recta y = x
distances = [math.sqrt((fpr[i])**2 + (1 - tpr[i])**2) for i in range(len(fpr))]
print(distances)

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
rows1, columns1 = GermanCredit.grupo_h.shape
rows2, columns1 = GermanCredit.grupo_m.shape
print("filas grupo hombres:", rows1)
print("filas grupo mujeres:", rows2)


#CALCULAMOS PROBABILIDADES
p1_dt = gsdec_tree.predict_proba(GermanCredit.grupo_h) #Calcula las probabilidades de los grupos
p2_dt = gsdec_tree.predict_proba(GermanCredit.grupo_m)

#-------PARIDAD-------
umbral_hp, umbral_mp =Fairness.umbralparidad(GermanCredit.grupo_h, GermanCredit.grupo_m, p1_dt, p2_dt, min_umbral)
print("umbra h:",umbral_hp, "umbral m:", umbral_mp)

#columna de prob que si reincidan
ind1 = p1_dt[:,1]>umbral_hp
ind2 = p2_dt[:,1]>umbral_mp

#concatenar las indices
indices = np.concatenate((ind1,ind2))

#grupos de los si reincidentes
grupo1 = GermanCredit.grupo_h[ind1, :]
grupo2 = GermanCredit.grupo_m[ind2, :]

#calculo de las filas
rows1, columns1 = grupo1.shape
rows2, columns2 = grupo2.shape
print("filas grupo hombres que superan el umbral:", rows1)
print("filas grupo mujeres que superan el umbral:", rows2)

p_y = Fairness.predictumbral(indices, GermanCredit.valid_yaux) #con ese umbral calculamos la prediccion
print('Accuracy: %.3f' % accuracy_score(p_y, GermanCredit.valid_yaux))
#vemos su matriz de confusion
cm = confusion_matrix(p_y, GermanCredit.valid_yaux)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=gsdec_tree.classes_)
disp.plot()
plt.show()

#-------IGUALDAD DE PROBABILIDADES-------
umbral_hip, umbral_mip = Fairness.umbralprobabilidades(GermanCredit.grupo_h, GermanCredit.grupo_m, p1_dt, p2_dt, gsdec_tree, GermanCredit.valid_yaux1, GermanCredit.valid_yaux2, GermanCredit.valid_yaux,min_umbral)
print("umbra h:",umbral_hip, "umbral m:", umbral_mip)

ind1ip = p1_dt[:,1]>umbral_hip #columna de prob que si reincidan
ind2ip = p2_dt[:,1]>umbral_mip

indicesig = np.concatenate((ind1ip,ind2ip))

p_yip = Fairness.predictumbral(indicesig, GermanCredit.valid_yaux) #con ese umbral calculamos la prediccion
print('Accuracy: %.3f' % accuracy_score(p_yip, GermanCredit.valid_yaux))
#vemos su matriz de confusion
cm = confusion_matrix(p_yip, GermanCredit.valid_yaux)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=gsdec_tree.classes_)
disp.plot()
plt.show()

#-------IGUALDAD DE OPORTUNIDADES-------
umbral_hio, umbral_mio = Fairness.umbraloportunidades(GermanCredit.grupo_h, GermanCredit.grupo_m, p1_dt, p2_dt, gsdec_tree, GermanCredit.valid_yaux1, GermanCredit.valid_yaux2, GermanCredit.valid_yaux, min_umbral)
print("umbra h:",umbral_hio, "umbral m:", umbral_mio)

ind1io = p1_dt[:,1]>umbral_hio #columna de prob que si reincidan
ind2io = p2_dt[:,1]>umbral_mio

indicesio = np.concatenate((ind1io,ind2io))

p_yio = Fairness.predictumbral(indicesio, GermanCredit.valid_yaux) #con ese umbral calculamos la prediccion
print('Accuracy: %.3f' % accuracy_score(p_yio, GermanCredit.valid_yaux))
#vemos su matriz de confusion
cm = confusion_matrix(p_yio, GermanCredit.valid_yaux)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=gsdec_tree.classes_)
disp.plot()
plt.show()

#-------IMPACTO DESIGUAL-------
umbral_hid, umbral_mid = Fairness.umbralimpactodesigual(GermanCredit.grupo_h, GermanCredit.grupo_m, p1_dt, p2_dt, gsdec_tree, GermanCredit.valid_yaux1, GermanCredit.valid_yaux2, GermanCredit.valid_yaux, min_umbral)
print("umbra h:",umbral_hid, "umbral m:", umbral_mid)

ind1id = p1_dt[:,1]>umbral_hid #columna de prob que si reincidan
ind2id = p2_dt[:,1]>umbral_mid

indicesid = np.concatenate((ind1id,ind2id))

p_yid = Fairness.predictumbral(indicesid, GermanCredit.valid_yaux) #con ese umbral calculamos la prediccion
print('Accuracy: %.3f' % accuracy_score(p_yid, GermanCredit.valid_yaux))
#vemos su matriz de confusion
cm = confusion_matrix(p_yid, GermanCredit.valid_yaux)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=gsdec_tree.classes_)
disp.plot()
plt.show()
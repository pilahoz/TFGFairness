import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


def predictumbral(indices,valid_yaux):
    p_y = list()
    # hago un for hasta el len de valid_y y voy añadiendo 1 si true y 0 si false para cambiar la prediccion
    for i in range(0, len(valid_yaux)):
        if indices[i]:
            p_y.append(1)
        else:
            p_y.append(0)
    return p_y


def umbralparidad(grupo_nf, grupo_f, p1_hat, p2_hat, min_umbral):
    umbral_nf = min_umbral
    umbral_f = min_umbral

    prob1 = p1_hat[:, 1]  # coger los positivos
    prob2 = p2_hat[:, 1]

    ind1 = prob1 > umbral_nf  # indices que sean mayor que el umbral
    ind2 = prob2 > umbral_f

    grupo1 = grupo_nf[ind1, :]  # coger los mayores que el umbral
    grupo2 = grupo_f[ind2, :]

    rows1, columns1 = grupo1.shape  # contar las filas
    rows2, columns2 = grupo2.shape

    paux = prob1[ind1]  # coger los que sean mayores que el umbral para no hacer sumas innecesarias
    paux = np.sort(paux)  # ordenarlos de menor a mayor
    paux = np.diff(paux)  # array con las diferencias entre los umbrales

    countp = 0

    for p in paux:
        if (rows1 > rows2):
            umbral_nf = p + umbral_nf  # sumando las diferencias para ir incrementando el umbral
            indaux = prob1 > umbral_nf  # indices de los que son mayores del umbral
            grupoaux = grupo_nf[indaux, :]  # nuevo grupo
            rows1 = len(grupoaux)  # cantidad de los que superan el umbral
            countp = countp + 1  # contador para ver su posicion
    # HACER MEDIA ENTRE EL ÚLTIMO Y EL SIGUIENTE (PARA QUE SEA PARA TODOS)
    if (len(paux) - 1) > countp:
        siguiente = umbral_nf + paux[countp + 1]  # media entre el que corta el if y su siguiente
        umbral_nf = (umbral_nf + siguiente) / 2

    return umbral_nf, umbral_f


def umbralprobabilidades(grupo_nf, grupo_f, p1_hat, p2_hat, gslog_reg, valid_yaux1, valid_yaux2, valid_yaux, min_umbral):

    umbral_nf = min_umbral
    umbral_f = min_umbral

    prob1 = p1_hat[:, 1] # coger los positivos
    prob2 = p2_hat[:, 1]

    ind1 = prob1 > umbral_nf # índices que sean mayor que el umbral
    ind2 = prob2 > umbral_f

    grupo1 = grupo_nf[ind1, :] # coger los mayores que el umbral
    grupo2 = grupo_f[ind2, :]

    op1valid_y = valid_yaux1[ind1]  # de la columna y los correspondientes al grupo1
    prob1pred_y = gslog_reg.predict(grupo1)  # predicción para la matriz del grupo1
    prob1cm = confusion_matrix(op1valid_y, prob1pred_y)  # Matriz de confusion1

    TN0, FP0, FN0, TP0 = prob1cm.ravel() # valores de matriz de confusion
    TPR0 = TP0 / (TP0 + FN0)
    FPR0 = FP0 / (FP0 + TN0)
    print(TN0, FP0, FN0, TP0 )


    op2valid_y = valid_yaux2[ind2] # de la columna y los correspondientes al grupo2
    prob2pred_y = gslog_reg.predict(grupo2)  # predicción para la matriz del grupo2
    prob2cm = confusion_matrix(op2valid_y, prob2pred_y) # Matriz de confusion2

    TN1, FP1, FN1, TP1 = prob2cm.ravel() # valores de matriz de confusion2
    TPR1 = TP1 / (TP1 + FN1)
    FPR1 = FP1 / (FP1 + TN1)

    paux = prob1[ind1] # probabilidad de los reindicidentes
    paux = np.sort(paux) # probabilidades ordenadas de menor a mayor
    paux = np.diff(paux) # array con las diferencias entre las probabilidades

    difTPR = list() #lista con las diferencias de TPR
    difFPR = list() #lista con las diferencias de FPR
    arrayumbrales = list() #list con las sumas de las p (umbrales)

    #se hace un for donde se crea un array con las diferencias entre TPR y FPR y escojo la que tenga la menor
    if TPR0 != TPR1 and FPR0 != FPR1:
        for p in paux:
            newumb = p + umbral_nf # nuevo umbral
            arrayumbrales.append(newumb) #se introduce el umbral en el array de umbrales
            umbral_nf = newumb #se actualiza el umbral
            indaux = prob1 > umbral_nf #indices donde la probabilidad sea mayor que el umbral

            indices = np.concatenate((indaux, ind2)) #se concatena los nuevos indices de umbral_nf con los de noaa

            p_y = predictumbral(indices,valid_yaux) #con ese umbral calculamos la prediccion
            probauxcm = confusion_matrix(p_y, valid_yaux) #se calcula la CM

            TN0, FP0, FN0, TP0 = probauxcm.ravel()  #se calculan TPR y FPR nuevos
            TPR0 = TP0 / (TP0 + FN0)
            FPR0 = FP0 / (FP0 + TN0)

            difT = TPR0 - TPR1 # se calculan las diferencias, ya que sea 0 es muy complicado
            difF = FPR0 - FPR1

            difTPR.append(difT) # se introducen las diferencias en los arrays
            difFPR.append(difF)

        idxTPR = np.abs(difTPR).argmin()  # Obtenemos el índice del valor absoluto más pequeño
        idxFPR = np.abs(difFPR).argmin()

        nmenorTPR = difTPR[idxTPR]  # Obtenemos el valor absoluto mas pequeño
        nmenorFPR = difFPR[idxFPR]


        # dar prioridad a uno que a otro para decidir que umbral coger
        if nmenorTPR < nmenorFPR:
            umbral_nf = arrayumbrales[idxTPR] # cogemos el valor en sí
            if (len(difTPR) - 1) != idxTPR:
                siguiente = umbral_nf + difTPR[idxTPR + 1]  # media entre el que corta el if y su siguiente
                umbral_nf = (umbral_nf + siguiente) / 2
        else:
            umbral_nf = arrayumbrales[idxFPR]
            if (len(difTPR) - 1) != idxFPR:
                siguiente = umbral_nf + difTPR[idxFPR + 1]  # media entre el que corta el if y su siguiente
                umbral_nf = (umbral_nf + siguiente) / 2


    return umbral_nf, umbral_f


def umbraloportunidades(grupo_nf, grupo_f, p1_hat, p2_hat, gslog_reg, valid_yaux1, valid_yaux2,valid_yaux, min_umbral):

    umbral_nf = min_umbral
    umbral_f = min_umbral

    prob1 = p1_hat[:, 1] # coger los positivos
    prob2 = p2_hat[:, 1]

    ind1 = prob1 > umbral_nf # índices que sean mayor que el umbral
    ind2 = prob2 > umbral_f

    grupo1 = grupo_nf[ind1, :] # coger los mayores que el umbral
    grupo2 = grupo_f[ind2, :]

    op1valid_y = valid_yaux1[ind1] # de la columna y los correspondientes al grupo1
    prob1pred_y = gslog_reg.predict(grupo1)  # predicción para la matriz del grupo1
    prob1cm = confusion_matrix(op1valid_y, prob1pred_y) # Matriz de confusion1

    TN0, FP0, FN0, TP0 = prob1cm.ravel() # valores de matriz de confusion
    TPR0 = TP0 / (TP0 + FN0)

    op2valid_y = valid_yaux2[ind2] # de la columna y los correspondientes al grupo2
    prob2pred_y = gslog_reg.predict(grupo2) # predicción para la matriz del grupo2
    prob2cm = confusion_matrix(op2valid_y, prob2pred_y)  # Matriz de confusion2

    TN1, FP1, FN1, TP1 = prob2cm.ravel() # valores de matriz de confusion2
    TPR1 = TP1 / (TP1 + FN1)

    paux = prob1[ind1] # probabilidad de los reindicidentes
    paux = np.sort(paux) # probabilidades ordenadas de menor a mayor
    paux = np.diff(paux) # array con las diferencias entre las probabilidades

    difTPR = [] #lista con las diferencias de TPR
    arrayumbrales = [] #list con las sumas de las p (umbrales)

    # Again, in practice, we use a cutoff to give some leeway.
    # This definition is supposed to represent the legal concept of disparate impact.
    # In the US there is a legal precedent to set the cutoff to 0.8.
    # That is the PPP for the unprivileged group must not be less than 80% of that of the privileged group.
    Cutoff = 0.8

    # se hace un for donde se crea un array con las diferencias de TPR y escojo la que tenga la menor
    if (TPR0 != TPR1) or (TPR1 - TPR0 > Cutoff) or (TPR0 / TPR1 < Cutoff):
        for p in paux:
            newumb = p + umbral_nf # nuevo umbral
            arrayumbrales.append(newumb) # se introduce el umbral en el array de umbrales
            umbral_nf = newumb # se actualiza el umbral
            indaux = prob1 > umbral_nf # indices donde la probabilidad sea mayor que el umbral

            indices = np.concatenate((indaux, ind2))  # se concatena los nuevos indices de umbral_nf con los de noaa

            p_y = predictumbral(indices, valid_yaux)  # con ese umbral calculamos la prediccion
            probauxcm = confusion_matrix(p_y, valid_yaux)  # se calcula la CM

            TN0, FP0, FN0, TP0 = probauxcm.ravel() #se calculan TPR
            TPR0 = TP0 / (TP0 + FN0)

            difT = TPR0 - TPR1 # se calculan las diferencias, ya que sea 0 es muy complicado
            difTPR.append(difT)  # se introducen las diferencias en los arrays

        idxTPR = np.abs(difTPR).argmin()  # Obtenemos el índice del valor absoluto más pequeño

        umbral_nf = arrayumbrales[idxTPR] #actualizamos umbral_nf con el valor

        if (len(difTPR) - 1) != idxTPR:
            siguiente = umbral_nf + difTPR[idxTPR + 1]  # media entre el que corta el if y su siguiente
            umbral_nf = (umbral_nf + siguiente) / 2


    return umbral_nf, umbral_f


def umbralimpactodesigual(grupo_nf, grupo_f, p1_hat, p2_hat, gslog_reg, valid_yaux1, valid_yaux2, valid_yaux, min_umbral):

    umbral_nf = min_umbral
    umbral_f = min_umbral

    prob1 = p1_hat[:, 1] # coger los positivos
    prob2 = p2_hat[:, 1]

    ind1 = prob1 > umbral_nf # índices que sean mayor que el umbral
    ind2 = prob2 > umbral_f

    grupo1 = grupo_nf[ind1, :] # coger los mayores que el umbral
    grupo2 = grupo_f[ind2, :]

    op1valid_y = valid_yaux1[ind1] # de la columna y los correspondientes al grupo1
    prob1pred_y = gslog_reg.predict(grupo1) # predicción para la matriz del grupo1
    prob1cm = confusion_matrix(op1valid_y, prob1pred_y) # Matriz de confusion1

    TN0, FP0, FN0, TP0 = prob1cm.ravel() # valores de matriz de confusion
    N0 = TP0 + FP0 + FN0 + TN0
    PPP0 = (TP0 + FP0) / N0

    op2valid_y = valid_yaux2[ind2] # de la columna y los correspondientes al grupo2
    prob2pred_y = gslog_reg.predict(grupo2) # predicción para la matriz del grupo2
    prob2cm = confusion_matrix(op2valid_y, prob2pred_y)  # Matriz de confusion2

    TN1, FP1, FN1, TP1 = prob2cm.ravel() # valores de matriz de confusion2
    N1 = TP1 + FP1 + FN1 + TN1
    PPP1 = (TP1 + FP1) / N1

    paux = prob1[ind1] # probabilidad de los reindicidentes
    paux = np.sort(paux) # probabilidades ordenadas de menor a mayor
    paux = np.diff(paux) # array con las diferencias entre las probabilidades

    difPPP = [] #lista con las diferencias de PPP
    arrayumbrales = [] #list con las sumas de las p (umbrales)

    # That is the PPP for the unprivileged group must not be less than 80% of that of the privileged group.
    Cutoff = 0.8
    # se hace un for donde se crea un array con las diferencias de PPP y escojo la que tenga la menor
    if (PPP0 != PPP1) or (PPP0 / PPP1 < Cutoff):

        for p in paux:
            newumb = p + umbral_nf # nuevo umbral
            arrayumbrales.append(newumb)# se introduce el umbral en el array de umbrales
            umbral_nf = newumb # se actualiza el umbral
            indaux = prob1 > umbral_nf  # índices donde la probabilidad sea mayor que el umbral

            indices = np.concatenate((indaux, ind2))  # se concatena los nuevos indices de umbral_nf con los de noaa

            p_y = predictumbral(indices, valid_yaux)  # con ese umbral calculamos la prediccion
            probauxcm = confusion_matrix(p_y, valid_yaux)  # se calcula la CM

            TN0, FP0, FN0, TP0 = probauxcm.ravel()
            N0 = TP0 + FP0 + FN0 + TN0
            PPP0 = (TP0 + FP0) / N0

            difP = PPP0 - PPP1
            difPPP.append(difP)

        idxPPP = np.abs(difPPP).argmin()  # Obtenemos el índice del valor absoluto más pequeño
        umbral_nf = arrayumbrales[idxPPP]

        if (len(difPPP) - 1) != idxPPP:
            siguiente = umbral_nf + difPPP[idxPPP + 1]  # media entre el que corta el if y su siguiente
            umbral_nf = (umbral_nf + siguiente) / 2



    return umbral_nf, umbral_f
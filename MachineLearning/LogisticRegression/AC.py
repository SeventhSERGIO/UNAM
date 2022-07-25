from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from math import inf
import seaborn as sn

def normalizeData(data):
    if type(data) is np.ndarray:
        return (data - data.min(axis=0))/(data.max(axis=0)-data.min(axis=0))
    elif type(data) is pd.core.frame.DataFrame:
        return (data - data.min())/(data.max()-data.min())
    else:
        print('Tipo de datos no valido')
    
def fold_i_of_k(dataset, i, k):
    n = len(dataset)
    trainSet = [dataset[0:n*(i-1)//k], dataset[n*i//k:]]
    testSet = dataset[n*(i-1)//k:n*i//k]
    if type(dataset) is np.ndarray:
        return np.concatenate(trainSet, axis=0), testSet 
    elif type(dataset) is pd.core.frame.DataFrame:
        return pd.concat(trainSet), testSet
    else:
        print('Tipo de datos no valido')
        
def train_test_split(inputs, outputs, testSize):
    n = len(inputs)
    m = len(outputs)
    
    if n == m:
        X_test = inputs[:int(n*testSize)]
        X_train = inputs[int(n*testSize):]
        y_test = outputs[:int(m*testSize)]
        y_train = outputs[int(m*testSize):]

    return X_train, X_test, y_train, y_test


def getLearningCurves(model, inputs, outputs, *, batch=1, **training):
    
    '''Return numpy arrays with the errors of the training set (with different instances) 
    and errors of the validation set'''
    
    # Se separan los datos de validacion" 
    X_train, X_val, y_train, y_val = train_test_split(inputs, outputs, testSize=0.2)
    train_errors, val_errors = [], []

    # Numero total de instancias de entrenamiento 
    n = len(X_train)

    if 'normalize' in training:
        normalize = training['normalize']
    else:
        normalize = False
        
    # Se realiza el entrenamiento utilizando el modelo con diferente tamano de instancias
    for m in range(1,n,batch):
        
        # Entrana el modelo con un numero definido de instancias y se le pasan los parametros
        model.train(X_train[:m], y_train[:m], **training)
        # Se obtienen las predicciones tanto del entrenamiento como de la validacion
        y_train_predict = model.predict(X_train[:m], normalize=normalize)
        y_val_predict = model.predict(X_val, normalize=normalize)
        # Se agrega el error generado con cada numero de instancias
        train_errors.append(meanSquareError(y_train_predict,y_train[:m]))
        val_errors.append(meanSquareError(y_val_predict,y_val))
        
    return np.sqrt(train_errors), np.sqrt(val_errors)
        
def meanSquareError(y_hat,y):
    return float(np.square(np.subtract(y_hat, y)).mean())

def crossValidation(model, inputs, outputs, k, **training):

    # Arreglos para almacenar cada costo generado
    cv_costs = np.zeros(k)

    # Se inicializa el valor del mejor costo
    bestCost = inf

    # Se hace el entrenamiento para cada k-fold
    for i in range(1,k+1):
        # Se obtiene la parte de entrenamiento y validacion
        X_cv_train, X_cv_test = fold_i_of_k(inputs,i,k)
        y_cv_train, y_cv_test = fold_i_of_k(outputs,i,k)

        # Se esntrena el modelo con cada uno de los folds
        model.train(X_cv_train,y_cv_train,**training)
        
        if 'normalize'in training:
            y_hat = model.predict(X_cv_test,normalize=training['normalize'])
        else:
            y_hat = model.predict(X_cv_test)

        currentCost = meanSquareError(y_hat,y_cv_test)
        cv_costs[i-1] = currentCost

        if currentCost < bestCost:
            bestCoeff = model.coeff
            bestCost = currentCost
    

    return np.sqrt(cv_costs), np.sqrt(cv_costs).mean(), np.sqrt(cv_costs).std(), bestCoeff

def getConfusionMatrixValues(real,predictions,threshold):
    # Se hacen arreglos de una dimension
    real = np.array(real).flatten()
    predictions = np.array(predictions).flatten()
    # Se inicializan las variables de conteo
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    # Se comparan los valores reales y las predicciones
    for y,predict in zip(real,predictions):
        if predict >= threshold:
            if y:
                TP += 1
            else:
                FP += 1
        else:
            if y:
                FN += 1
            else:
                TN += 1
    return TP,TN,FP,FN

def clasificationMetrics(real,predictions,*,threshold=0.5):
    # Se obtiene el total de positivos y negativos
    P = np.count_nonzero(real)
    N = len(real) - P 
    TP,TN,FP,FN = getConfusionMatrixValues(real,predictions,threshold)
    # Se crea la matriz de confusion
    confusionMatrix = np.array([[TP,FN],[FP,TN]])
    # Grafica la matriz de confusion
    plt.figure(figsize=(12,8))
    g = sn.heatmap(confusionMatrix, annot=True, fmt='.5g', cmap='viridis', 
                   annot_kws={"fontsize":20}, cbar=False)
    g.set_xticklabels(['Male (1)','Female (0)'],fontdict={'size':18})
    g.set_yticklabels(['Male (1)','Female (0)'],fontdict={'size':18})
    g.set_ylabel('Actual Condition',fontdict={'size':18, 'weight': 'bold'})
    g.set_xlabel('Predicted Condition',fontdict={'size':18,'weight': 'bold'})
    plt.show()
    
    # Se calculan las Metricas de clasificacion
    # Error
    ERR = (FP+FN)/(FP+FN+TP+TN)
    # Exactitud
    ACC = (TP+TN)/(FP+FN+TP+TN)
    # Sensibilidad (Tasa de Verdaderos Positivos)
    TPR = TP/P
    # Especificidad
    TNR = TN/N
    # Tasa de falsos positivos
    FPR = FP/N
    # Precision
    PRE = TP/(TP+FP)
    # F1-Score
    F1 = 2*((PRE*TPR)/(PRE+TPR))
    
    # Se crean las tablas de las metricas
    counts = pd.DataFrame([P+N,P,N,TP,TN,FP,FN],columns=['counts'], 
                      index=['Total Population', 'Total Positives (P)', 
                             'Total Negatives (N)', 'True Positives (TP)', 
                             'True Negatives (TN)', 'False Positives (FP)', 
                             'False Negatives (FN)'])
    metrics = pd.DataFrame([ERR,ACC,TPR,TNR,FPR,PRE,F1],columns=['value'], 
                          index=['Error (ERR)', 'Accuracy (ACC)', 'Sensitivity (TPR)', 
                                 'Specificity (TNR)','False Positive Rate (FPR)', 
                                 'Precision (PRE)', 'F1-Score (F1)'])
    return counts, metrics
        
class LinearRegression():
    
    def __init__(self):
        self.coeff = None
        self.costHistory = None
        self.coeffHistory = None
        print('Linear Regressor Created')
    
    def meanSquareErrorPrime(self,X,y_hat,y,m):
        return (2/m)* X.T.dot(np.subtract(y_hat, y))
    
    def predict(self,inputs,*,normalize=False, addOnes=True):
        
        # Si normalize es True, se normalizan las entradas
        if normalize:
            inputs = normalizeData(inputs)
        
        # Se convierten las entradas a numpy arrays
        X = np.array(inputs)
        
        # agrega los unos si la bandera esta activa
        if addOnes:
            X = np.c_[np.ones(X.shape[0]), X]
        
        return X.dot(self.coeff)
    
    def gradientDescent(self, X, y, eta, epochs):
        
        # Numero de instancias
        m = len(y)
        
        # Arreglo donde se almacena el historial de los costos
        self.costHistory = np.zeros(epochs)
        # Arreglo donde se almacena el historial de los coeficientes
        self.coeffHistory = np.zeros((epochs,X.shape[1]))

        # Modifica el valor de los parametros haciendo pequeños pasos
        for epoch in range(epochs):
            # Hace una prediccion con las entradas
            y_hat = self.predict(X, addOnes=False)
            # Calcula todos los gradientes 
            gradients = self.meanSquareErrorPrime(X,y_hat,y,m)
            # Calcula los nuevos coeficientes a partir de la eta y los gradientes
            self.coeff = self.coeff - eta * gradients
            
            # Almacena el historial de los coeficientes en cada iteracion
            self.coeffHistory[epoch,:] = self.coeff.T
            # Almacena el historial de los costos en cada iteracion
            self.costHistory[epoch] = meanSquareError(y_hat,y)
        
        self.costHistory = np.sqrt(self.costHistory)
        self.cost = np.sqrt(meanSquareError(y_hat,y))
    
    def train(self, inputs, outputs, *, normalize=False, eta = 0.1, epochs=1000):
        
        # Si normalize es True, se normalizan las entradas
        if normalize:
            inputs = normalizeData(inputs)
            
        # Se convierten las entradas a numpy arrays
        X = np.array(inputs)
        X = np.c_[np.ones(X.shape[0]), X]
        
        # Se convierten las salidas a numpy arrays
        y = np.array(outputs)
        
        # Se remoldea el vector a mx1
        if len(y.shape) < 2:
            y = y.reshape(-1,1)
            
        # Se inicializan los coeficientes del regresor
        self.coeff = np.random.rand(X.shape[1],1)   
        
        # Se entrea el regresor por medio del descenso del gradiente
        self.gradientDescent(X, y, eta, epochs)
        

                
class LogisticRegression():
    
    def __init__(self):
        self.coeff = None
        print('Logistic Regressor Created')
    
    def sigmoid(self, z):
        return 1/(1+np.exp(-z))
    
    def costFunction(self,p_hat, y):
        return (-y*np.log(p_hat)-(1-y)*np.log(1-p_hat)).mean()
    
    def predict(self,inputs,*,normalize=False, addOnes=True):
        
        # Si normalize es True, se normalizan las entradas
        if normalize:
            inputs = normalizeData(inputs)
        
        # Se convierten las entradas a numpy arrays
        X = np.array(inputs)
        
        # agrega los unos si la bandera esta activa
        if addOnes:
            X = np.c_[np.ones(X.shape[0]), X]
        
        return self.sigmoid(X.dot(self.coeff))
    
    def gradientDescent(self, X, y, eta, epochs):
        
        # Numero de instancias
        m = len(y)
        # Arreglo donde se almacena el historial de los costos
        self.costHistory = np.zeros(epochs)
        # Arreglo donde se almacena el historial de los coeficientes
        self.coeffHistory = np.zeros((epochs,X.shape[1]))

        # Modifica el valor de los parametros haciendo pequeños pasos
        for epoch in range(epochs):
            # Hace una prediccion con las entradas
            p_hat = self.predict(X, addOnes=False)
            # Calcula todos los gradientes 
            gradients = 1/m * np.dot(X.T,(p_hat - y))
            
            
            # Calcula los nuevos coeficientes a partir de la eta y los gradientes
            self.coeff = self.coeff - eta * gradients
            
            # Almacena el historial de los coeficientes en cada iteracion
            self.coeffHistory[epoch,:] = self.coeff.T
            
            # Almacena el historial de los costos en cada iteracion
            self.costHistory[epoch] = self.costFunction(p_hat,y)
            #print(f'epoch:{epoch} cost:{self.costHistory[epoch]}')
        
        self.cost = self.costHistory[-1]
    
    def train(self, inputs, targets, *, normalize=False, eta=0.1, epochs=1000):
        
        # Si normalize es True, se normalizan las entradas
        if normalize:
            inputs = normalizeData(inputs)
        
        # Se convierten las entradas a numpy arrays
        X = np.array(inputs)
        X = np.c_[np.ones(X.shape[0]), X]
        
        # Se convierten las salidas a numpy arrays
        y = np.array(targets)
        
        # Se remoldea el vector a mx1
        if len(y.shape) < 2:
            y = y.reshape(-1,1)
            
        # Se inicializan los coeficientes del regresor
        self.coeff = np.random.rand(X.shape[1],1)  
        
        # Se entrea el regresor por medio del descenso del gradiente
        self.gradientDescent(X, y, eta, epochs)
        
        
    





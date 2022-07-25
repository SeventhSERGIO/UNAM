from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from math import inf

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
        
class LinearRegression():
    
    def __init__(self):
        self.coeff = None
        self.costHistory = None
        self.coeffHistory = None
        print('Linear Ressor Created')
    
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
        

                


        
        
    





from matplotlib import pyplot as plt
from matplotlib import colors
import numpy as np
import pandas as pd
import seaborn as sn

def getDatasets(N=1000, R=1, *, limFunction, squareSide=1, max1Iterations=50000):
    # Variables para contar el numero de instancias dentro de cada clase
    inside, outside = 0, 0
    dataPoints = []
    # Se genera la iteracion hasta llenar los datos en cada clase o
    # agotar el numero max1imo de iteraciones
    while max1Iterations and inside < N or outside < N:
        # Se generan dos puntos aleatorios dentro del area del cuadrado
        x = np.random.uniform(-squareSide,squareSide)
        y = np.random.uniform(-squareSide,squareSide)
        # Si los puntos generados se encuentran dentro de region marcada
        # por la funcion limite pertenecen a la clase 0
        if limFunction(x,y,R):
            if inside < N:
                dataPoints.append([x,y,0])
                inside += 1
        # De lo contrario pertenecen a la clase 1
        else: 
            if outside < N:
                dataPoints.append([x,y,1])
                outside += 1 
        max1Iterations -= 1
    return pd.DataFrame(np.array(dataPoints),columns=['X','Y','Class'])

def getConfusionMatrixValues(real,predictions):
    # Se inicializan las variables de conteo
    TP,TN,FP,FN = 0,0,0,0
    # Se comparan los valores reales y las predicciones
    for y,predict in zip(real,predictions):
        if predict:
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

def getSVM(dataset, model, *, supportVectors=False, cmapReal=['tab:red','gray'], markers=['o','D'],
           errorColor=(101/255,110/255,242/255), class0='Class 0', class1='Class 1', cmapPred=['yellow','gray']):
    # Se obtienen los vectores para el entrenamiento
    X = np.array(dataset.iloc[:,:2])
    y = np.array(dataset.iloc[:,2])
    n0 = np.argwhere(y==0)
    n1 = np.argwhere(y==1)
    # Se entrena la SVM con el conjunto de datos
    model.fit(X,y)
    # Se abtiene las figuras a utilizar
    fig = plt.figure(figsize=(20,8))
    fig.tight_layout(h_pad=50.0)
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.set_facecolor((230/255,236/255,245/255))
    ax1.grid(c='white',which='both')
    ax1.tick_params(axis='x', colors='k',labelsize=14, grid_color='white')
    ax1.tick_params(axis='y', colors='k',labelsize=14, grid_color='white')
    ax1.tick_params(tick1On=False)
    ax1.xaxis.label.set_color('k')
    ax1.yaxis.label.set_color('k')
    ax1.set_xlabel('x', fontdict={'size':18})
    ax1.set_ylabel('y', fontdict={'size':18})
    ax1.set_axisbelow(True)
    ax1.spines['bottom'].set_color('white')
    ax1.spines['top'].set_color('white')
    ax1.spines['right'].set_color('white')
    ax1.spines['left'].set_color('white')
    # Se grafican los datos reales divididos por clase
    ax1.scatter(X[n0, 0], X[n0, 1], marker=markers[0], edgecolor='k', linewidths=0.3, color=cmapReal[0], label='Real '+ class0)
    ax1.scatter(X[n1, 0], X[n1, 1], marker=markers[1], edgecolor='k', linewidths=0.3, color=cmapReal[1], label='Real '+ class1)
    # Se obtienen las regines de prediccion
    x_min, x_max1 = X[:, 0].min()-0.1, X[:, 0].max()+0.1
    y_min, y_max1 = X[:, 1].min()-0.1, X[:, 1].max()+0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max1, 0.01), np.arange(y_min, y_max1, 0.01))
    z = model.predict(np.c_[xx.ravel(),yy.ravel()]).reshape(xx.shape)
    cf = ax1.contourf(xx, yy, z, cmap=colors.ListedColormap(cmapPred), alpha=0.3)
    cbar = fig.colorbar(cf, ax=ax1)
    cbar.set_ticks([0.25,0.75])
    cbar.set_ticklabels([class0,class1],fontsize=16)
    cbar.set_label("Predicction",fontsize=18, rotation=270, fontweight="bold", labelpad=-1)
    cbar.ax.tick_params(size=0)
    # Se obtiene los vectores que se clasificaron mal y se grafican
    y_hat = model.predict(X)
    error_y = np.nonzero(y != y_hat)
    ax1.scatter(X[error_y, 0], X[error_y, 1], facecolors='none', edgecolors=errorColor, linewidths=1, label='Error')
    # Si la bandera esta activa obtiene los vectores de soporte y se grafican
    if supportVectors:
        ax1.scatter(x=model.support_vectors_[:,0],y=model.support_vectors_[:,1], 
                   color='k', label='Support Vectors', marker='x', alpha=0.6)
    # Se configura las legendas
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=5, fontsize=12)
    # Se obtiene la exactitud y se muestra
    acc = model.score(X,y)
    if model.kernel != 'poly':
        ax1.set_title(f'SVM kernel: {model.kernel}\n Mean accuracy:{round(acc,3)}',fontsize=16,fontweight="bold")
    else:
        ax1.set_title(f'SVM kernel: {model.kernel} degree: {model.degree}\n Mean accuracy:{round(acc,3)}',fontsize=16,fontweight="bold")
    # Se obtiene la matriz de confusion
    TP,TN,FP,FN = getConfusionMatrixValues(y,y_hat)
    confusionMatrix = np.array([[TP,FN],[FP,TN]])
    # Grafica la matriz de confusion
    sn.heatmap(confusionMatrix, annot=True, fmt='.5g', annot_kws={"fontsize":20}, cbar=False, ax=ax2)
    ax2.set_xticklabels([f'{class1} (1)',f'{class0} (0)'],fontdict={'size':18})
    ax2.set_yticklabels([f'{class1} (1)',f'{class0} (0)'],fontdict={'size':18})
    ax2.set_ylabel('Actual Condition',fontdict={'size':18, 'weight': 'bold'})
    ax2.set_xlabel('Predicted Condition',fontdict={'size':18,'weight': 'bold'})
    ax2.set_title('Confusion matrix', fontsize=16, fontweight='bold')
    return model
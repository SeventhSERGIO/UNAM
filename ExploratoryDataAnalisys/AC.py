from matplotlib import pyplot as plt
import numpy as np

specialPurple = (101/255,110/255,242/255)
specialGreen = (92/255,201/255,154/255)
specialRed = (221/255,96/255,70/255)
otherGreen = (71/255,148/255,147/255)

def stylizePlot(ax, xlabel, ylabel):
    ax.set_facecolor((230/255,236/255,245/255))
    ax.grid(c='white',which='both')
    ax.tick_params(axis='x', colors='k',labelsize=14, grid_color='white')
    ax.tick_params(axis='y', colors='k',labelsize=14, grid_color='white')
    ax.tick_params(tick1On=False)
    ax.xaxis.label.set_color('k')
    ax.yaxis.label.set_color('k')
    ax.set_xlabel(xlabel, fontdict={'size':13, 'weight':'bold'}, labelpad=10)
    ax.set_ylabel(ylabel, fontdict={'size':13, 'weight':'bold'}, labelpad=10)
    ax.set_axisbelow(True)
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.spines['left'].set_color('white')
    return ax

def createFigures(*, figsize=(6,6), xlabel='', ylabel='', n_axes=1, columns=3):
    if n_axes <= columns:
        rows = 1
        cols = n_axes
    else:
        rows = int(np.ceil(n_axes/columns))
        cols = columns
    
    figsize = (figsize[0]*cols, figsize[1]*rows)
    
    fig = plt.figure(figsize=figsize)
    fig.tight_layout(pad=30)
    
    axes = []
    for ax in range(n_axes):
        ax = fig.add_subplot(rows,cols,ax+1)
        axes.append(stylizePlot(ax, xlabel, ylabel))
        
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, 
                        top=0.9, wspace=0.4, hspace=0.4)
    
    if len(axes) == 1:
        return fig, axes[0]
    else:
        return fig, axes
    
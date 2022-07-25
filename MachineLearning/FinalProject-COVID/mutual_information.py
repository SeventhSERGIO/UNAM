from collections import Counter
from scipy.stats import entropy
import numpy as np
import pandas as pd
import math 

def conditional_entropy(x,y):
    # Entropy of x given y
    # Count the elements on the container
    y_counter = Counter(y)
    # Count the elements on both containers
    xy_counter = Counter(list(zip(x,y)))
    # Total instances
    total_occurrences = sum(y_counter.values())
    # Sum the entropy of every occurence in the counter
    entropy = 0
    for xy in xy_counter.keys():
        p_xy = xy_counter[xy] / total_occurrences
        p_y = y_counter[xy[1]] / total_occurrences
        entropy += p_xy * math.log(p_y/p_xy)
    return entropy

def univariate_entropy(data, base=None):
    # Get the frecuency of each apperance
    _, f = np.unique(data, return_counts=True)
    # return the entropy with the selected base (default e)
    return entropy(f) if base is None else entropy(f, base=base)

def mutal_information(x,y):
    h = univariate_entropy(x)
    # Get the mutal information between two random variables
    return (h - conditional_entropy(x,y))/h

def mutal_info_from_frame(data, *, columns=None, normalize=False):
    # Get the mutal information between every atribute of a frame
    # If columns are not define take all the columns to compare
    if columns is None:
        columns = data.columns
    # Create a frame with the atributes to compare
    temp = pd.DataFrame(index=columns, columns=data.columns, dtype=float)
    for row in temp.index:
        for column in temp:
            temp.loc[row,column] = mutal_information(data[row], data[column])
    # Normalize and the return the values
    if normalize:
        return (temp - temp.min())/(temp.max()-temp.min())
    else:
        return temp
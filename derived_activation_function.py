from activation_function import sigmoid , softmax
from Activation import Activation
import numpy as np

def derived_linear(x):
    return 1

def derived_RELU(x):
    return 0 if x < 0 else 1

def derived_sigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

def derived_softmax(x, target):
    return x if 1 != target else -(1-x)

def derived(activation : Activation, x, target=None):
    if (activation == Activation.RELU):
        return derived_RELU(x)
    elif (activation == Activation.linear):
        return derived_linear(x)
    elif (activation == Activation.sigmoid):
        return derived_sigmoid(x)
    elif (activation == Activation.softmax):
        return derived_softmax(x, target)

def derived_RELU_matrix(matriks):
    matriks = np.array(matriks)
    #print(matriks)
    result = matriks.copy()
    for i in range(len(matriks)):
        for j in range(len(matriks[i])):
            #print(matriks[i][j],end=' ')
            #print(result[i][j],end=' ')
            result[i][j] = derived_RELU(matriks[i][j])
        #print()
    return result

#Test
# size = 5
# random_matrix = np.random.randint(-10,10,(size,size))
# print(random_matrix)
# result_matrix = derived_RELU_matrix(random_matrix)
# print(result_matrix)

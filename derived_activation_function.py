from activation_function import sigmoid , softmax
from Activation import Activation


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

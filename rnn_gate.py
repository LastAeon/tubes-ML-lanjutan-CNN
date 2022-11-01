from activation_function import sigmoid
import numpy as np

X = [
    [1, 2],
    [.5, 3]
]

hprev = 0
Cprev = 0

Uf = [.7, .45]
Ui = [.95, .8]
Uc = [.45, .25]
Uo = [.6, .4]

Wf, bf = .1, .15
Wi, bi = .8, .65
Wc, bc = .15, .2
Wo, bo = .25, .1

def calculate_gate(u: list[float], x: list[float], w: float, h: float, b:float):
    return sigmoid(np.dot(u, x) + w * h + b)

def calculate_forgot(uf: list[float], x: list[float], wf: float, hprev: float, bf:float):
    return calculate_gate(uf, x, wf, hprev, bf)

def calculate_input(ui: list[float], x: list[float], wi: float, hprev: float, bi:float):
    return calculate_gate(ui, x, wi, hprev, bi)

print(calculate_gate(Uf, X[0], Wf, hprev, bf))
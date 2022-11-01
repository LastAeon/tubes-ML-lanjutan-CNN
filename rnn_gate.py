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

#Gate sigmoid
def calculate_gate(u: list[float], x: list[float], w: float, h: float, b:float):
    return sigmoid(np.dot(u, x) + w * h + b)

#Forgot gate
def calculate_forgot(uf: list[float], x: list[float], wf: float, hprev: float, bf:float):
    return calculate_gate(uf, x, wf, hprev, bf)

#Input gate
def calculate_input(ui: list[float], x: list[float], wi: float, hprev: float, bi:float):
    return calculate_gate(ui, x, wi, hprev, bi)

#Calculate candidate
def calculate_candidate(uc: list[float], x: list[float], wc: float, hprev: float, bc:float):
    return np.tanh(np.dot(uc, x) + wc * hprev + bc)

#Cell state
def calculate_cell(ft: float, cprev:float, it: float, candidate:float):
    return ft*cprev + it*candidate

#Output gate
def calculate_output(uo: list[float], x: list[float], wo: float, hprev: float, bo:float):
    return calculate_gate(uo,x,wo,hprev,bo)

def calculate_hidden(ot:float, ct:float):
    return ot * np.tanh(ct)

#print(calculate_gate(Uf, X[0], Wf, hprev, bf))
#TEST
#X timestep 1
# x1 = X[0]
# ft = calculate_forgot(Uf, x1, Wf, hprev, bf)
# print("ft: ",ft)
# it = calculate_input(Ui, x1, Wi, hprev, bi)
# print("it: ",it)
# candidate = calculate_candidate(Uc, x1, Wc, hprev, bc)
# print("C~: ",candidate)
# ct = calculate_cell(ft, Cprev, it, candidate)
# print("ct: ",ct)
# ot = calculate_output(Uo, x1, Wo, hprev, bo)
# print("ot: ",ot)
# ht = calculate_hidden(ot,ct)
# print("ht: ",ht)
# print("Timestep 1 ct {} dan ht {}".format(ct,ht))
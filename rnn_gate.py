from activation_function import sigmoid
import numpy as np

class Cell:
    #rand_initialize(bool): parameter randomized 
    #U(list of matrix): kumpulan matriks U
    #f: forgot gate weight & bias
    #i: input gate weight & bias
    #c: cell state weight & bias
    #o: output gate weight & bias
    output = []

    def hitungValue(self,x, cprev, hprev, verbose=False):
        return self.calculate_timestep(x, cprev, hprev, verbose)

    def printNeuron(self):
        print(self.U, self.f, self.i, self.c, self.o)
    def __init__(self, rand_initialize, U, f, i, c, o):
        #Not implemented input size

        if(rand_initialize == False):
            self.U = U
            self.f = f
            self.i = i
            self.c = c
            self.o = o

            self.Uf = U[0]
            self.Ui = U[1]
            self.Uc = U[2]
            self.Uo = U[3]
            
            self.Wf = f[0]
            self.bf = f[1]
            self.Wi = i[0]
            self.bi = i[1]
            self.Wc = c[0]
            self.bc = c[1]
            self.Wo = o[0]
            self.bo = o[1]

            # self.cprev = 0
            # self.hprev = 0
        else:
            print("Not Implemented")
    
    def calculate_timestep(self,x, cprev=0, hprev=0, verbose=False):
        ft = self.calculate_forgot(Uf, x, Wf, hprev, bf)
        it = self.calculate_input(Ui, x, Wi, hprev, bi)
        candidate = self.calculate_candidate(Uc, x, Wc, hprev, bc)
        ct = self.calculate_cell(ft, cprev, it, candidate)
        ot = self.calculate_output(Uo, x, Wo, hprev, bo)
        ht = self.calculate_hidden(ot,ct)

        if(verbose):
            print("ht: ",ht)
            print("ft: ",ft)
            print("it: ",it)
            print("c~: ",candidate)
            print("ct: ",ct)
            print("ot: ",ot)
            print("Timestep ct {} dan ht {}".format(ct,ht))
        
        # self.cprev = ct
        # self.hprev = ht
        return ct,ht

    #Gate sigmoid
    def calculate_gate(self,u: list[float], x: list[float], w: float, h: float, b:float):
        return sigmoid(np.dot(u, x) + w * h + b)

    #Forgot gate
    def calculate_forgot(self,uf: list[float], x: list[float], wf: float, hprev: float, bf:float):
        return self.calculate_gate(uf, x, wf, hprev, bf)

    #Input gate
    def calculate_input(self,ui: list[float], x: list[float], wi: float, hprev: float, bi:float):
        return self.calculate_gate(ui, x, wi, hprev, bi)

    #Calculate candidate
    def calculate_candidate(self,uc: list[float], x: list[float], wc: float, hprev: float, bc:float):
        return np.tanh(np.dot(uc, x) + wc * hprev + bc)

    #Cell state
    def calculate_cell(self,ft: float, cprev:float, it: float, candidate:float):
        return ft*cprev + it*candidate

    #Output gate
    def calculate_output(self,uo: list[float], x: list[float], wo: float, hprev: float, bo:float):
        return self.calculate_gate(uo,x,wo,hprev,bo)

    def calculate_hidden(self,ot:float, ct:float):
        return ot * np.tanh(ct)

#TEST class
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

testCell = Cell(False,[Uf,Ui,Uc,Uo],[Wf,bf],[Wi,bi],[Wc,bc],[Wo,bo])
for x in X:
    hasil = testCell.calculate_timestep(x,verbose=True)
    print(hasil)
testCell.printNeuron()
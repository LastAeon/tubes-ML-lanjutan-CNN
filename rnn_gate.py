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
        print("U", self.U)
        print("f", self.f)
        print("i", self.i)
        print("c", self.c)
        print("o", self.o)

    def __init__(self, rand_initialize, U, f, i, c, o, input_length=0):
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
            self.Uf = np.random.rand(input_length).tolist()
            self.Ui = np.random.rand(input_length).tolist()
            self.Uc = np.random.rand(input_length).tolist()
            self.Uo = np.random.rand(input_length).tolist()
            
            self.Wf = np.random.rand()
            self.bf = np.random.rand()
            self.Wi = np.random.rand()
            self.bi = np.random.rand()
            self.Wc = np.random.rand()
            self.bc = np.random.rand()
            self.Wo = np.random.rand()
            self.bo = np.random.rand()

            self.U = [self.Uf,self.Ui,self.Uc,self.Uo]
            self.f = [self.Wf,self.bf]
            self.i = [self.Wi,self.bi]
            self.c = [self.Wc,self.bc]
            self.o = [self.Wo,self.bo]
    
    def calculate_timestep(self,x, cprev=0, hprev=0, verbose=False):
        ft = self.calculate_forgot(self.Uf, x, self.Wf, hprev, self.bf)
        it = self.calculate_input(self.Ui, x, self.Wi, hprev, self.bi)
        candidate = self.calculate_candidate(self.Uc, x, self.Wc, hprev, self.bc)
        ct = self.calculate_cell(ft, cprev, it, candidate)
        ot = self.calculate_output(self.Uo, x, self.Wo, hprev, self.bo)
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
    def calculate_gate(self,u, x, w: float, h: float, b:float):
        return sigmoid(np.dot(u, x) + w * h + b)

    #Forgot gate
    def calculate_forgot(self,uf, x, wf: float, hprev: float, bf:float):
        return self.calculate_gate(uf, x, wf, hprev, bf)

    #Input gate
    def calculate_input(self,ui, x, wi: float, hprev: float, bi:float):
        return self.calculate_gate(ui, x, wi, hprev, bi)

    #Calculate candidate
    def calculate_candidate(self,uc, x, wc: float, hprev: float, bc:float):
        return np.tanh(np.dot(uc, x) + wc * hprev + bc)

    #Cell state
    def calculate_cell(self,ft: float, cprev:float, it: float, candidate:float):
        return ft*cprev + it*candidate

    #Output gate
    def calculate_output(self,uo, x, wo: float, hprev: float, bo:float):
        return self.calculate_gate(uo,x,wo,hprev,bo)

    def calculate_hidden(self,ot:float, ct:float):
        return ot * np.tanh(ct)

#TEST class
# X = [
#     [1, 2],
#     [.5, 3]
# ]

# hprev = 0
# Cprev = 0

# Uf = [.7, .45]
# Ui = [.95, .8]
# Uc = [.45, .25]
# Uo = [.6, .4]

# Wf, bf = .1, .15
# Wi, bi = .8, .65
# Wc, bc = .15, .2
# Wo, bo = .25, .1

# testCell = Cell(False,[Uf,Ui,Uc,Uo],[Wf,bf],[Wi,bi],[Wc,bc],[Wo,bo])
# for x in X:
#     hasil = testCell.calculate_timestep(x,verbose=True)
#     print(hasil)
# testCell.printNeuron()

# X = [
#     [1, 2],
#     [.5, 3],
#     [1, 2],
#     [.5, 3],
# ]

# testCell = Cell(True,0,0,0,0,0,len(X[0]))
# for x in X:
#     hasil = testCell.calculate_timestep(x,verbose=True)
#     print(hasil)
# testCell.printNeuron()
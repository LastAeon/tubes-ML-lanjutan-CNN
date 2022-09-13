from Activation import Activation
import derived_activation_function as derived

class Neuron:
    bobot = None
    netValue = None

    def __init__(self, bobot):
        self.bobot = bobot
        self.netValue = None
        self.errorFactor = None
        self.deltaWeight = [0 for i in range(len(bobot))]
    # fungsi hitungValue menghitung net dari penjumlahan antara perkalian bobot dengan input
    def hitungValue(self,input):
        self.netValue = sum(x * y for x, y in zip(self.bobot, input))
    def getNetValue(self):
        return self.netValue
    def printNeuron(self):
        print(self.bobot)
    def updateBobot(self, learningRate):
        
        for i in range(len(self.bobot)):
            self.bobot[i] -= learningRate * self.deltaWeight[i]
            self.deltaWeight[i] = 0
        
    def getBobot(self):
        return self.bobot
    def getError(self):
        return self.errorFactor
    
    
    #Hitung error output
    #NOTES
    #Output ada di layer?
    #Target belum ada
    #Fungsi turunan dari output
    def calculateErrorOut(self,output,activation, prevOutput, target):
        #NOTES
        #-log(pk)
        #ini masukin errorFactor aja?
        # print("target: ", target)
        # print('output ', output)
        if(activation == Activation.softmax):
            self.errorFactor =  derived.derived(activation, output, target)
        else:
            self.errorFactor = -(target-output)*derived.derived(activation, output)
        
        # print("target ", target, "output ", output)
        # print("errorFactor", self.errorFactor)
        self.deltaWeight[0] += self.errorFactor
        for i in range (1, len(self.bobot)):
            #JANGAN LUPA ADA BIAS
            self.deltaWeight[i] += self.errorFactor * prevOutput[i-1]

        # print(self.errorFactor)

    #Hitung hidden error
    #Hitung gradient 
    #X merupakan input
    #prevOutput : array berisi output dari neuron pada layer sebelumnya yang menjadi input neuron ini
    #output : output dari neuron ini, diambil dari layer
    def calculateHiddenError(self,output,activation, prevOutput, nextWeight, nextError):
        #nextWeight : array berisi bobot dari tiap neuron pada layer setelahnya yang menerima input dari neuron ini
         #nextError : array error dari setiap neuron pada layer berikutnya
        errorOutput = 0
        # print("next weight: ",nextWeight)
        # print("next error: ", nextError)
        for i in range(len(nextWeight)):
            errorOutput += nextWeight[i] * nextError[i]
        # print("error Output", errorOutput)
        # if(activation == Activation.softmax):
        #     self.errorFactor = derived.derived_softmax(output,target) 
        if(activation == Activation.linear):
            self.errorFactor = errorOutput* derived.derived_sigmoid(output)
        elif(activation == Activation.RELU):
            self.errorFactor = errorOutput* derived.derived_RELU(output)
        elif(activation == Activation.sigmoid):
            self.errorFactor = errorOutput* derived.derived_linear(output)

        # print("deltaWeight sebelum ",self.deltaWeight)
        # print("error factor", self.errorFactor)
        self.deltaWeight[0] += self.errorFactor
        for i in range (1, len(self.bobot)):
            #JANGAN LUPA ADA BIAS
            self.deltaWeight[i] += self.errorFactor * prevOutput[i-1]
        # print("deltaWeight sesudah ",self.deltaWeight)

if (__name__ == "__main__"):
    # p1 = Neuron()
    # p1.hitungValue([1,2,2],[1,3,5])
    # print(p1.getNetValue())
    pass    
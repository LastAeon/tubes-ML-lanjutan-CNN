from rnn_gate import Cell
from Activation import Activation
from activation_function import linear, RELU, sigmoid, softmax 

class Layer_LSTM:
    def __init__(self):
        self.neurons = []
        self.output = []
        self.aktivasi = 'LSTM'
        self.bias = 1
        self.errorFactor = 0
        self.isOutput = False

    def addNeuron(self, cell: Cell):
        self.neurons.append(cell)
    
    def setToOutput(self):
        self.isOutput = True


    def hitungOutput(self, layerInput):
        self.output.clear()
        layerInput.insert(0, self.bias)
        ct = 0
        ht = 0
        i = 0
        for n in self.neurons:
            ct, ht = n.hitungValue(layerInput[i], ct, ht)
            i += 1
        #     n.hitungValue(layerInput)
        #     if (self.aktivasi == Activation.linear):
        #         self.output.append(linear(n.getNetValue()))
        #     elif(self.aktivasi == Activation.RELU):
        #         self.output.append(RELU(n.getNetValue()))
        #     elif(self.aktivasi == Activation.sigmoid):
        #         self.output.append(sigmoid(n.getNetValue()))
        #     elif(self.aktivasi == Activation.softmax):
        #         self.output.append(n.getNetValue())
        # if (self.aktivasi == Activation.softmax):
        #     self.output = softmax(self.output)
        self.output.append(ht)

    def printLayer(self):
        print("Layer:  LSTM")
        i = 1
        for n in self.neurons:
            print(f"Cell {i}:", end=" ")
            n.printNeuron()
            i += 1

    def emptyOutput(self):
        self.output.clear()
    
    def getOutput(self):
        output = []
        for i in self.output:
            output.append(i)
        return output
    
#     def getError(self):
#         error = []
#         for n in self.neurons:
#             error.append(n.getError())
#         return error

#     def getNeuronWeight(self):
#         weights = []
#         for n in self.neurons:
#             weights.append(n.getBobot())
#         return weights

#     def updateBobot(self, learningRate):
#         for n in self.neurons:
#             n.updateBobot(learningRate)
   
#     def computeDeltaBobot(self, prevLayer, nextLayer = None, target = None):
#         #cek hidden atau output layer
#         #for all neuron calculateHiddenError atau calculateErrorOut
#         if (self.isOutput):
#             # print("is output layer")
#             for i in range(len(self.neurons)):
#                 self.neurons[i].calculateErrorOut(self.output[i], self.aktivasi, prevLayer.getOutput(), target[i])
#         else:
#             # print("not output layer")
#             Weights = nextLayer.getNeuronWeight()
#             for i in range(len(self.neurons)):
#                 nextWeight = []
#                 for j in range(len(Weights)):
#                     nextWeight.append(Weights[j][i])
#                 self.neurons[i].calculateHiddenError(self.output[i], self.aktivasi, prevLayer.getOutput(), nextWeight, nextLayer.getError())
#         # return prevBobot + learningRate * (target - output) * input

# # p1 = Layer(Activation.linear)
# # p1.computeHiddenError(2,0)
# # print(p1.computeDeltaBobot(0,0.3,-1,0,1))
# # print(p1.getError())

# # p2 = Layer(Activation.RELU)
# # p2.computeHiddenError(1,1)
# # print(p2.getError())

# # p3 = Layer(Activation.sigmoid)
# # p3.computeHiddenError(1,1)
# # print(p3.getError())

# # p4 = Layer(Activation.softmax)
# # p4.computeHiddenError(0,1.83*10**-15)
# # print(p4.getError())



                

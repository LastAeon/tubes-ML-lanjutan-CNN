import numpy as np
from Layer import Layer
from Activation import Activation
from Neuron import Neuron


activation = {"sigmoid": Activation.sigmoid, "linear": Activation.linear,"RELU": Activation.RELU,"softmax": Activation.softmax }

class FFNN:
    layer_list = []
    learning_rate = 0
    cumulative_error = 99999999999999
    expected_output =[]

    def __init__(self, filename):
        #destroy object
        self.layer_list = []
        #baca dari file
        with open(filename) as reader:
            filecontent = reader.read()
            lines = filecontent.split("\n")
            for i in range(int(lines.pop(0))):
                layer_info = lines.pop(0).split(" ")
                layer = Layer(activation[layer_info[0]])
                for j in range(int(layer_info[-1])):
                    neuron = Neuron(list(map(float, lines.pop(0).split(" "))))
                    layer.addNeuron(neuron)
                self.addLayer(layer)
        self.layer_list[-1].setToOutput()
        return

    def addLayer(self, layer : Layer):
        self.layer_list.append(layer)

    def predict(self, input):
        output = input.copy()
        for i in self.layer_list:
            i.hitungOutput(output)
            output = i.getOutput()
        # print("output ", output)
        return output

    def predictBatch(self, input_array):
        output_array = [] 
        for j in range(len(input_array)):
            input = input_array[j]
            output = input.copy()
            for i in self.layer_list:
                i.hitungOutput(output)
                output = i.getOutput()
            output_array.append(output)
        return output_array

    def printModel(self):
        j = 1
        for i in self.layer_list:
            print(f"Layer {j}:")
            i.printLayer()
            j += 1
    
    # set parameter that used for backward passing
    def setBackwardParameter(self, ex_output, learn_rate):
        self.expected_output = ex_output
        self.learning_rate = learn_rate

    # compute output cost for error margin
    def computeCost(self, output, target):
        error = 0
        for i in range(len(target)):
            if (self.layer_list[-1].aktivasi == Activation.softmax):
                for j in range(len(target[i])):
                    if(target[i][j] == 1):
                        error += -np.log(output[i][j])
            else:
                for j in range(len(target[i])):
                    error += (target[i][j]-output[i][j])**2/2
        return error
    
    # compute and update weight in model  
    def adjustWeight(self):
        for layer in self.layer_list:
            # ubah bobot
            layer.updateBobot(self.learning_rate)

    def computeError(self, entryIndex):
        for i in range(len(self.layer_list)-1, 0, -1):
            # itung error factor
            if(i == len(self.layer_list)-1):
                self.layer_list[i].computeDeltaBobot(self.layer_list[i-1], target = self.expected_output[entryIndex])
            else:
                self.layer_list[i].computeDeltaBobot(self.layer_list[i-1], nextLayer = self.layer_list[i+1])

    def backward(self, batch_size, error_threshold, max_iteration, input):
        # split batch
        batch = [input[i:i+batch_size] for i in range(0,len(input),batch_size)]


        # execute
        iter = 0
        cumulative_error = 0
        notDone = True
        while(notDone):
            if(iter == max_iteration):
                print("error: ", cumulative_error)
            epoch_result = []
            entry_index = 0
            for current_batch in batch:
                for entri in current_batch:
                    # print(entry_index, entri)
                    epoch_result.append(self.predict(entri))
                    # add input layer
                    layer_input = Layer(activation["linear"])
                    layer_input.output = entri.copy()
                    self.layer_list.insert(0, layer_input)
                    self.computeError(entry_index)
                    self.layer_list.pop(0)
                    # self.printModel()
                    entry_index += 1
                self.adjustWeight()
                # self.printModel()
            cumulative_error = self.computeCost(epoch_result, self.expected_output)
            iter += 1
            if(not(cumulative_error > error_threshold and iter < max_iteration)):
                notDone = False
        self.printModel()
        print("iter: ", iter)
        print("error: ", cumulative_error)
        self.cumulative_error = cumulative_error





# Kelas FFNN
# Atribut:
# List of Layer
# Method:
# readConfig (Baca konfigurasi) -> konstruktor
# addLayer
# predict (menerima input batch)

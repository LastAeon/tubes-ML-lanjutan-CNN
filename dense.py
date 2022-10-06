from FFNN import FFNN
from util import flatten
import numpy as np

class Dense:
    def __init__(self, filename):
        self.type = 'dense layer'
        self.ffnn = FFNN(filename)
    
    def predict_set_of_matrix(self, set_of_matrix):
        return self.ffnn.predict(flatten(set_of_matrix))

    def init_backpropagation(self, expected_output, learning_rate):
        self.ffnn.setBackwardParameter(expected_output, learning_rate)

    def backpropagation(self, set_of_matrix):
        original_shape = np.shape(set_of_matrix)
        # print('original_shape:', original_shape)
        layer_weight = [self.weight_in_layer(i) for i in range(len(self.ffnn.layer_list))]
        # print(layer_weight)
        self.ffnn.backward(1, 0, 1, [flatten(set_of_matrix)])
        derivativ_for_next_layer = self.binary_cross_entropy()

        for i in range(len(self.ffnn.layer_list)-1, -1, -1):
            # print('derivativ_for_next_layer:', derivativ_for_next_layer)
            # print('layer_weight:', layer_weight[i])
            derivativ_for_next_layer = np.dot(derivativ_for_next_layer, layer_weight[i])

        derivativ_for_next_layer = np.reshape(derivativ_for_next_layer, original_shape)
        # print('dense result:', derivativ_for_next_layer)
        return derivativ_for_next_layer

    def binary_cross_entropy(self):
        temp_array = []
        for neuron in self.ffnn.layer_list[-1].neurons:
            # print("bobot output:", neuron.bobot)
            # print('binary_cross_entropy', neuron.errorFactor)
            temp_array.append(neuron.errorFactor)
        return temp_array
    
    def weight_in_layer(self, idx):
        temp_matrix = []
        for neuron in self.ffnn.layer_list[idx].neurons:
            temp_array = []
            for bobot in neuron.bobot:
                temp_array.append(bobot)
            temp_array.pop()
            temp_matrix.append(temp_array)
        return temp_matrix

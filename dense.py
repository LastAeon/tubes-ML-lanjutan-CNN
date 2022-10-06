from FFNN import FFNN
from util import flatten

class Dense:
    def __init__(self, filename):
        self.type = 'dense layer'
        self.ffnn = FFNN(filename)
    
    def predict_set_of_matrix(self, set_of_matrix):
        return self.ffnn.predict(flatten(set_of_matrix))

    def init_backpropagation(self, expected_output, learning_rate):
        self.ffnn.setBackwardParameter(expected_output, learning_rate)

    def backpropagation(self, set_of_matrix):
        self.ffnn.backward(1, 0, 1, [flatten(set_of_matrix)])

    def binary_cross_entropy(self):
        temp_array = []
        for neuron in self.ffnn.layer_list[-1].neurons:
            # print("bobot output:", neuron.bobot)
            # print('binary_cross_entropy', neuron.errorFactor)
            temp_array.append(neuron.errorFactor)
        return temp_array

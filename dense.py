from FFNN import FFNN
from flatten import flatten

class Dense:
    def __init__(self, filename):
        self.ffnn = FFNN(filename)
    
    def predict_set_of_matrix(self, set_of_matrix):
        return self.ffnn.predict(flatten(set_of_matrix))

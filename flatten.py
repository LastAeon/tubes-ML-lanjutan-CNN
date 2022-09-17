import numpy as np

def flatten(set_of_matrix):
    resuting_array = []
    for matrix in set_of_matrix:
        temp_matrix = np.matrix(matrix).flatten().tolist()[0]
        resuting_array += temp_matrix
    
    return resuting_array
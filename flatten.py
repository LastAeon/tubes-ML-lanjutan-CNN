import numpy as np

def flatten(set_of_matrix):
    resuting_array = []
    for matrix in set_of_matrix:
        temp_matrix = np.matrix(matrix).flatten().tolist()[0]
        resuting_array += temp_matrix
    
    return resuting_array

set_of_matrix = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
print(flatten(set_of_matrix))
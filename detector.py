import numpy as np

def detector_elm(value: float):
    return max(0, value)

def detector(matrix: list[list[float]]): # Pokoknya, matrix tipenya array of array of float
    new_matrix = []
    for row in matrix:
        new_row = []
        for elm in row:
            new_elm = detector_elm(elm)
            new_row.append(new_elm)
        new_matrix.append(new_row)
    return new_matrix

def derive_detector_output(matrix: list[list[float]]):
    new_matrix = []
    for row in matrix:
        new_row = []
        for elm in row:
            new_elm = 1 if elm > 0 else 0
            new_row.append(new_elm)
        new_matrix.append(new_row)
    return new_matrix

def mult_2_matrix(m1: list[list[float]], m2: list[list[float]]):
    if (len(m1) != len(m2) or len(m1[0]) != len(m2[0])):
        raise Exception('mult_2_matrix: 2 matrices has different dimension')
    
    col_len = len(m1)
    row_len = len(m1[0])

    new_matrix = []
    for row_idx in range (col_len):
        new_row = []
        for elm_idx in range (row_len):
            new_row.append(m1[row_idx][elm_idx] * m2[row_idx][elm_idx])
        new_matrix.append(new_row)
    return new_matrix

class DetectorStep:
    def __init__(self):
        self.type = 'detector layer'
        self.prev_output = None
        return
    def hitungOutput(self, input_matrixes):
        # print("detector:", np.shape(input_matrixes))
        output_list = []
        for matrix in input_matrixes:
            detected = detector(matrix)
            output_list.append(detected)
        self.prev_output = output_list
        return output_list
    def backpropagation(self, front_layer_output):
        output_list = []
        for i in range (len(self.prev_output)):
            derived = derive_detector_output(self.prev_output[i])
            multed_derived = mult_2_matrix(derived, front_layer_output[i])
            output_list.append(multed_derived)
        return output_list
def detector_elm(value: float):
    return max(0, value)

def detector(matrix): # Pokoknya, matrix tipenya array of array of float
    new_matrix = []
    for row in matrix:
        new_row = []
        for elm in row:
            new_elm = detector_elm(elm)
            new_row.append(new_elm)
        new_matrix.append(new_row)
    return new_matrix

class DetectorStep:
    def __init__(self):
        return
    def hitungOutput(self, input_matrixes):
        output_list = []
        for matrix in input_matrixes:
            output_list.append(detector(matrix))
        return output_list
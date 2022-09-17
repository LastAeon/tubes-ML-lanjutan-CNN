import numpy as np

# KET: unoptimized, matrix hasilnya harus di-transpose dlu baru bener
def pooling(raw_matrix, kernel_y, kernel_x, stride: int, mode: str= 'max' ):
    output = []

    max_y = raw_matrix.__len__()
    max_x = len(raw_matrix[0])
    
    first_x = 0
    first_y = 0
    while first_x < max_x+1-kernel_x:
        output_y = []
        first_y = 0
        while first_y < max_y+1-kernel_y:
            output_elm = 0
            raw_elm_list = []
            for y in range (kernel_y):
                for x in range (kernel_x):
                    raw_elm_list.append(raw_matrix[first_y + y][first_x + x])
            
            if (mode == 'max'):
                output_elm = max(raw_elm_list)
            else :
                output_elm = sum(raw_elm_list) / len(raw_elm_list)
            
            output_y.append(output_elm)
            first_y += stride
        output.append(output_y)
        first_x += stride
    return transpose(output)

def transpose(matrix):
    return np.transpose(matrix)

def pooling_layer(raw_matrix_list, kernel_y, kernel_x, stride: int, mode: str= 'max'):
    output_list = []
    for matrix in raw_matrix_list:
        output_list.append(pooling(matrix, kernel_y, kernel_x, stride, mode))
    return output_list

class PoolingStep:
    def __init__(self, raw_matrix_list, kernel_y, kernel_x, stride: int, mode: str= 'max'):
        self.raw_matrix_list = raw_matrix_list
        self.kernel_y = kernel_y
        self.kernel_x = kernel_x
        self.stride = stride
        self.mode = mode

    def pooling(self):
        pooling_layer(self.raw_matrix_list, self.kernel_y, self.kernel_x, self.stride, self.mode)

# def trial():
#     matrix = [
#         [1,2,3,4],
#         [5,6,7,8],
#         [9,0,1,2],
#         [3,4,5,6]
#     ]

#     pooling_output = pooling(matrix, 2, 2, 1, 'max')
#     print(pooling_output)

# trial()
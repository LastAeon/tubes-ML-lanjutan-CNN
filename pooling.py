import numpy as np

# KET: unoptimized, matrix hasilnya harus di-transpose dlu baru bener
def pooling(raw_matrix, kernel_y, kernel_x, stride: int, mode: str= 'max' ):

    max_y = raw_matrix.__len__()
    max_x = len(raw_matrix[0])
    
    output = []
    # For old backprop: output_derivation = np.zeros((max_y, max_x))
    output_der = []

    first_x = 0
    first_y = 0

    # Loop thru y (rows)
    while first_y < max_y+1-kernel_y:
        output_y = []
        output_der_y = []
        first_x = 0

        # Loop thru x (cols)
        while first_x < max_x+1-kernel_x:
            output_elm = 0
            raw_elm_list = []

            # For max pooling
            if (mode == 'max'):
                max_elm_idx = (first_y, first_x)
                curr_max_elm = raw_matrix[first_y][first_x]
                
                for y in range (kernel_y):
                    for x in range (kernel_x):
                        challenger_elm = raw_matrix[first_y + y][first_x + x]
                        if (challenger_elm > curr_max_elm):
                            curr_max_elm = challenger_elm
                            max_elm_idx = (first_y + y, first_x + x)
                        # raw_elm_list.append(raw_matrix[first_y + y][first_x + x])
            
                output_elm = curr_max_elm # max(raw_elm_list)
                output_der_y.append(max_elm_idx)
                # For old backprop: output_derivation[max_elm_idx[0]][max_elm_idx[1]] += 1
                

            # For avg pooling
            else :
                kernel_area = kernel_y * kernel_x
                for y in range (kernel_y):
                    for x in range (kernel_x):
                        raw_elm_list.append(raw_matrix[first_y + y][first_x + x])
                        # For old backprop: output_derivation[first_y + y][first_x + x] += 1/kernel_area
            
                output_elm = sum(raw_elm_list) / len(raw_elm_list)
                output_der_y.append(0,0)
            
            output_y.append(output_elm)
            first_x += stride
        output.append(output_y)
        output_der.append(output_der_y)
        first_y += stride
    return output, output_der # transpose(output)

def transpose(matrix):
    return np.transpose(matrix)

def pooling_layer(raw_matrix_list, kernel_y, kernel_x, stride: int, mode: str= 'max'):
    output_list = []
    output_derivation_list = []
    for matrix in raw_matrix_list:
        pool_output, pool_der_output = pooling(matrix, kernel_y, kernel_x, stride, mode)
        output_list.append(pool_output)
        output_derivation_list.append(pool_der_output)
    return output_list, output_derivation_list

class PoolingStep:
    def __init__(self, kernel_y, kernel_x, stride: int, mode: str= 'max'):
        self.kernel_y = kernel_y
        self.kernel_x = kernel_x
        self.stride = stride
        self.mode = mode

        self.derived_output = None
        self.input = None

    def hitungOutput(self, input_matrixes):
        output, output_derv = pooling_layer(input_matrixes, self.kernel_y, self.kernel_x, self.stride, self.mode)
        self.derived_output = output_derv
        self.input = input_matrixes
        return output

    def hitungBack(self, front_layer_derv_list):
        output = []

        for mtx_idx in range (len(front_layer_derv_list)) :
            output_mtx = np.zeros((len(self.input[mtx_idx]), len(self.input[mtx_idx][0])))
            front_layer_derv = front_layer_derv_list[mtx_idx]
            max_location_mtx = self.derived_output[mtx_idx]

            # Read where max value is located
            for row_idx in range (len(front_layer_derv)):
                for col_idx in range (len(front_layer_derv[row_idx])):

                    value = front_layer_derv[row_idx][col_idx]

                    # If max
                    if (self.mode == 'max'):
                        max_location = max_location_mtx[row_idx][col_idx]
                        output_mtx[max_location[0]][max_location[1]] += value
                    
                    else: # If Avg
                        kernel_area = self.kernel_x * self.kernel_y
                        for y in range (self.kernel_y):
                            for x in range (self.kernel_x):
                                output_mtx[row_idx + y][col_idx + x] += value / kernel_area

            output.append(output_mtx)
        
        return output

        # else : # Avg
        #     max_y = self.output[0].__len__()
        #     max_x = len(self.output[0][0])
        #     kernel_y = self.kernel_y
        #     kernel_x = self.kernel_x
        #     stride = self.stride
        #     first_x = 0
        #     first_y = 0

        #     output_der = np.zeros((max_y, max_x))

        #     while first_y < max_y+1-kernel_y:
        #         first_x = 0

        #         # Loop thru x (cols)
        #         while first_x < max_x+1-kernel_x:
        #             kernel_area = kernel_y * kernel_x
        #             for y in range (kernel_y):
        #                 for x in range (kernel_x):
        #                     output_der[first_y + y][first_x + x] += 1/kernel_area
        #             first_x += stride
        #         first_y += stride
        
# def trial():
#     matrix = [
#         [
#             [1,2,3,4],
#             [5,6,7,8],
#             [9,0,1,2],
#             [3,4,5,6]
#         ]
#     ]

#     pooler = PoolingStep(2, 2, 1, 'max')

#     pooling_output = pooler.hitungOutput(matrix)
#     print("Pooling Output")
#     print(np.matrix(pooling_output[0]))
#     print()

#     front_derv_mtx = [
#         [
#             [1, 2, 3],
#             [4, 5, 6],
#             [7, 8, 9]
#         ]
#     ]

#     derv_output = pooler.hitungBack(front_derv_mtx)
#     print("Derv Output")
#     print(np.matrix(derv_output[0]))
#     print()

# trial()
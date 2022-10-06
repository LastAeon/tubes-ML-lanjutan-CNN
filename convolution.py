from math import floor
import numpy as np
from scipy import signal

class ConvolutionStep:
    def __init__(self, banyak_channel, input_pad, banyak_filter, filter_size, input_stride):
        #NOTE: input untuk h dan w 
        self.type = 'convolution layer'
        self.banyak_channel = banyak_channel
        self.input_pad = input_pad
        self.banyak_filter = banyak_filter
        self.filter_size = filter_size
        self.input_stride = input_stride
        self.kernel_matrixes = []
        self.kernel_matrixes_shape = (banyak_channel, banyak_filter, filter_size, filter_size)
        self.bias_shape = (banyak_filter)
        self.prev_delta_berat = 0
        self.prev_delta_bias = 0

        #Filter kernel matrixes & bias
        #Randomize weight
        self.kernel_matrixes = np.random.randint(-10,10,self.kernel_matrixes_shape)
        # self.kernel_matrixes = np.ones(self.kernel_matrixes_shape)
        #Bias 
        self.bias_array = np.random.randint(1,2, self.bias_shape)
        # self.bias_array = np.ones(self.bias_shape)


 
    def hitungOutput(self, input_matrixes):
        # Padding Input matrixes
        self.input_matrixes = input_matrixes
        self.input_matrixes_shape = (len(input_matrixes), len(input_matrixes[0]), len(input_matrixes[0][0]))
        input_matrixes_after_padding = []
        
        for matrix in self.input_matrixes:
            #Padding
            #input_matrixes.append(np.pad(random_matrix,input_pad, mode='empty').tolist())
            input_matrixes_after_padding.append(np.pad(matrix, self.input_pad, 'constant', constant_values=(0)))
        self.input_matrixes = input_matrixes_after_padding

        output=[]
        channel_idx = -1
        for matriks in self.input_matrixes:
            channel_idx += 1
            v_h= floor((len(matriks[0])-self.filter_size)/self.input_stride + 1)
            v_w = floor((len(matriks)-self.filter_size)/self.input_stride + 1)
            h = len(matriks[0])
            w = len(matriks)

            output_matrixes = []
            for kernel_idx in range(self.banyak_filter):
                kernel_matriks = self.kernel_matrixes[channel_idx][kernel_idx]
                v_w_counter = 0
                i = 0
                output_matrix=[]
                while v_w_counter < v_w:
                    v_h_counter = 0
                    j=0
                    output_row = []
                    while v_h_counter < v_h:
                        i_filter = 0
                        conv_sum = 0
                        #convolution
                        for i2 in range(i,i+self.filter_size):
                            j_filter = 0
                            for j2 in range(j,j+self.filter_size):
                                conv_sum += matriks[i2][j2] * kernel_matriks[i_filter][j_filter]
                                j_filter+=1
                            i_filter+=1
                        output_row.append(conv_sum)
                        j+=self.input_stride
                        v_h_counter+=1
                    output_matrix.append(output_row)
                    i+=self.input_stride
                    v_w_counter+=1
                output_matrixes.append(output_matrix)
            output_final_matrix = [[0 for i_create in range(v_h)] for j_create in range(v_w)]
            # print("output_final_matrix0:", np.shape(output_final_matrix))
            # #debug
            # print(self.bias_array)
            # print(output_matrixes)
            for _idx in range(len(output_matrixes)):
                _matrix = output_matrixes[_idx]
                for _i in range(len(_matrix)):  
                    for _j in range(len(_matrix[0])):
                        output_final_matrix[_i][_j] += _matrix[_i][_j] 
                        if(_idx == len(output_matrixes)-1):
                            output_final_matrix[_i][_j] += self.bias_array[kernel_idx]
            output.append(output_final_matrix)
            # print("output_row:", np.shape(output_row))
            # print("output_matrix:", np.shape(output_matrix))
            # print("output_matrixes:", np.shape(output_matrixes))
            # print("output_final_matrix:", np.shape(output_final_matrix))
            # print("output:", np.shape(output))
            # print("bias_array:", np.shape(self.bias_array))
        self.output = output
        return output

    def init_backpropagation(self, learning_rate, momentum):
        self.learning_rate = learning_rate
        self.momentum = momentum

    def backpropagation(self, prev_layer_matrix):
        delta_bias = np.array([np.sum(prev_layer_matrix[_]) for _ in range(self.banyak_filter)])
        delta_berat = np.zeros(self.kernel_matrixes_shape)
        derivativ_for_next_layer = np.zeros(self.input_matrixes_shape)
        # print('dimensi delta berat:', np.shape(delta_berat))
        # print('dimensi input_matrixes:', np.shape(self.input_matrixes))
        # print('dimensi prev_layer_matrix:', np.shape(prev_layer_matrix))

        for i in range(self.banyak_filter):
            for j in range(self.banyak_channel):
                # print(';, j:', i, j)
                # print(signal.correlate2d(self.input_matrixes[j], prev_layer_matrix[i], 'valid'))
                # print(np.shape(signal.correlate2d(self.input_matrixes[j], prev_layer_matrix[i], 'valid')))
                delta_berat[j, i] = signal.correlate2d(self.input_matrixes[j], prev_layer_matrix[i], 'valid')
                derivativ_for_next_layer[j] = signal.convolve2d(prev_layer_matrix[i], self.kernel_matrixes[j][i], 'full')
        
        self.kernel_matrixes = self.kernel_matrixes - self.learning_rate * delta_berat - self.momentum*self.prev_delta_berat
        self.bias_array = self.bias_array - self.learning_rate * delta_bias - self.momentum*self.prev_delta_bias

        self.prev_delta_berat = delta_berat
        self.prev_delta_bias = delta_bias

        return derivativ_for_next_layer

        
        


        
    

# def test():
#     #Ukuran input 
#     banyak_input = int(input("Input banyaknya matriks: ")) 
#     input_h, input_w = input("Ukuran matriks (h w): ").split()
#     input_h, input_w = int(input_h), int(input_w)
#     #Ukuran padding
#     input_pad = int(input("Input ukuran padding: ")) 
#     #Jumlah filter
#     banyak_filter = int(input("Input banyaknya filter: ")) 
#     #Ukuran filter
#     #Simetris
#     filter_size = int(input("Input ukuran filter (1 value): ")) 
#     #Ukuran stride
#     input_stride = int(input("Input ukuran stride: ")) 
#     print()
#     #Spatial size V=(w-f+2p/s)+1
#     V_width = (input_w-filter_size+2*input_pad)/input_stride + 1
#     V_height = (input_h-filter_size+2*input_pad)/input_stride + 1
#     print('Expected output {0} and {1}'.format(floor(V_width),floor(V_height)))

#     input_matrixes = []
#     for i in range(banyak_input):
#         #Randomize for testing 
#         input_matrixes.append(np.random.randint(0,10,(input_h,input_w)))

#     #Object
#     convolutionStepTest = ConvolutionStep(input_matrixes, input_pad, banyak_filter, filter_size,input_stride)

#     #Print for debug
#     print_matrix("Input matriks", convolutionStepTest.input_matrixes)
#     print_matrix("Kernel matriks", convolutionStepTest.kernel_matrixes)

#     #Convolution output
#     convolutionStepTest.convolution()
#     output = convolutionStepTest.output
#     print_matrix("Output matriks", output)

# #Debug matrix
# def print_matrix(header,matrixes):
#     print(header)
#     for x in matrixes:
#         for y in x:
#             print(y)
#         print("----------------")

# #Call test function
# #test()



from math import floor
import numpy as np

class ConvolutionStep:
    def __init__(self,input_matrixes, input_pad, banyak_filter, filter_size, input_stride):
        self.banyak_input = len(input_matrixes)
        #NOTE: input untuk h dan w 
        self.input_pad = (input_pad)
        self.banyak_filter = banyak_filter
        self.filter_size = filter_size
        self.input_stride = input_stride

        #Input matrixes
        self.input_matrixes = input_matrixes
        input_matrixes = []
        for matrix in self.input_matrixes:
            #Padding
            #input_matrixes.append(np.pad(random_matrix,input_pad, mode='empty').tolist())
            input_matrixes.append(np.pad(matrix,input_pad,'constant', constant_values=(0)))
        self.input_matrixes = input_matrixes

        #Filter kernel matrixes
        kernel_matrixes = []
        for i in range(banyak_filter):
            #Randomize weight
            random_matrix = np.random.randint(-10,10,(filter_size,filter_size))
            kernel_matrixes.append(random_matrix)
        self.kernel_matrixes = kernel_matrixes
 
    def convolution(self):
        output=[]
        for matriks in self.input_matrixes:
            v_h= floor((len(matriks[0])-self.filter_size)/self.input_stride + 1)
            v_w = floor((len(matriks)-self.filter_size)/self.input_stride + 1)
            h = len(matriks[0])-self.filter_size
            w = len(matriks)-self.filter_size

            output_matrixes = []
            for kernel_matriks in self.kernel_matrixes:
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
            output.append(output_matrixes)
        self.output = output

def test():
    #Ukuran input 
    banyak_input = int(input("Input banyaknya matriks: ")) 
    input_h, input_w = input("Ukuran matriks (h w): ").split()
    input_h, input_w = int(input_h), int(input_w)
    #Ukuran padding
    input_pad = int(input("Input ukuran padding: ")) 
    #Jumlah filter
    banyak_filter = int(input("Input banyaknya filter: ")) 
    #Ukuran filter
    #Simetris
    filter_size = int(input("Input ukuran filter (1 value): ")) 
    #Ukuran stride
    input_stride = int(input("Input ukuran stride: ")) 
    print()
    #Spatial size V=(w-f+2p/s)+1
    V_width = (input_w-filter_size+2*input_pad)/input_stride + 1
    V_height = (input_h-filter_size+2*input_pad)/input_stride + 1
    print('Expected output {0} and {1}'.format(floor(V_width),floor(V_height)))

    input_matrixes = []
    for i in range(banyak_input):
        #Randomize for testing 
        input_matrixes.append(np.random.randint(0,10,(input_h,input_w)))

    #Object
    convolutionStepTest = ConvolutionStep(input_matrixes, input_pad, banyak_filter, filter_size,input_stride)

    #Print for debug
    print_matrix("Input matriks", convolutionStepTest.input_matrixes)
    print_matrix("Kernel matriks", convolutionStepTest.kernel_matrixes)

    #Convolution output
    convolutionStepTest.convolution()
    output = convolutionStepTest.output
    for o_matrixes in output:
        print_matrix("Output matriks", o_matrixes)

#Debug matrix
def print_matrix(header,matrixes):
    print(header)
    for x in matrixes:
        for y in x:
            print(y)
        print("----------------")

#Call test function
#test()



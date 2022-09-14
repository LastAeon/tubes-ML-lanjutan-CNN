from math import floor
import numpy as np

class ConvolutionStep:
    def __init__(self, banyak_input, input_h, input_w, input_pad, banyak_filter, filter_size, input_stride):
        self.banyak_input = banyak_input
        #NOTE: input untuk h dan w 
        self.input_h = input_h
        self.input_w = input_w
        self.input_pad = input_pad
        self.banyak_filter = banyak_filter
        self.filter_size = filter_size
        self.input_stride = input_stride

        #Input matrixes
        input_matrixes = []
        for i in range(banyak_input):
            #Randomize for testing 
            random_matrix = np.random.randint(0,10,(input_h,input_w))
            #Padding
            #input_matrixes.append(np.pad(random_matrix,input_pad, mode='empty').tolist())
            input_matrixes.append(np.pad(random_matrix,input_pad,'constant', constant_values=(0)))
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
            i = 0
            output_matrixes = []
            output_matrix=[]
            for indeks_kernel in range(len(self.kernel_matrixes)):
                while i <  self.input_w+self.input_pad:
                    j=0
                    output_row = []
                    while j < self.input_h+self.input_pad:
                        i_filter = 0
                        conv_sum = 0
                        #convolution
                        for i2 in range(i,i+self.filter_size):
                            j_filter = 0
                            for j2 in range(j,j+self.filter_size):
                                conv_sum += matriks[i2][j2] * self.kernel_matrixes[indeks_kernel][i_filter][j_filter]
                                j_filter+=1
                            i_filter+=1
                        output_row.append(conv_sum)
                        j+=self.input_stride
                    output_matrix.append(output_row)
                    i+=self.input_stride
                output_matrixes.append(output_matrix)
            output.append(output_matrixes)
        self.output = output
        return output

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

    #Object
    convolutionStepTest = ConvolutionStep(banyak_input, input_h, input_w, input_pad, banyak_filter, filter_size,input_stride)

    #Print for debug
    print_matrix("Input matriks", convolutionStepTest.input_matrixes)
    print_matrix("Kernel matriks", convolutionStepTest.kernel_matrixes)

    #Convolution output
    output = convolutionStepTest.convolution()
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
test()



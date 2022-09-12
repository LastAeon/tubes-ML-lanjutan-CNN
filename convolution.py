from math import floor
import numpy as np

def main():
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

    #Input matrixes
    input_matrixes = []
    for i in range(banyak_input):
        #Randomize for testing 
        random_matrix = np.random.randint(0,10,(input_h,input_w))
        #Padding
        #input_matrixes.append(np.pad(random_matrix,input_pad, mode='empty').tolist())
        input_matrixes.append(np.pad(random_matrix,input_pad,'constant', constant_values=(0)))

    #Filter kernel matrixes
    kernel_matrixes = []
    for i in range(banyak_filter):
        #Randomize weight
        random_matrix = np.random.randint(-10,10,(filter_size,filter_size))
        kernel_matrixes.append(random_matrix)

    print_matrix("Input matriks", input_matrixes)
    print_matrix("Kernel matriks", kernel_matrixes)

    output=[]
    for matriks in input_matrixes:
        i = 0
        output_matrixes = []
        output_matrix=[]
        for indeks_kernel in range(len(kernel_matrixes)):
            while i <  input_w+input_pad:
                j=0
                output_row = []
                while j < input_h+input_pad:
                    i_filter = 0
                    conv_sum = 0
                    #convolusion
                    for i2 in range(i,i+filter_size):
                        j_filter = 0
                        for j2 in range(j,j+filter_size):
                            conv_sum += matriks[i2][j2] * kernel_matrixes[indeks_kernel][i_filter][j_filter]
                            j_filter+=1
                        i_filter+=1
                    output_row.append(conv_sum)
                    j+=input_stride
                output_matrix.append(output_row)
                i+=input_stride
            output_matrixes.append(output_matrix)
        output.append(output_matrixes)

    for o_matrixes in output:
        print_matrix("Output matriks", o_matrixes)

#Debug matrix
def print_matrix(header,matrixes):
    print(header)
    for x in matrixes:
        for y in x:
            print(y)
        print("----------------")

main()



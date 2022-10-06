# baris 70-75 handling konvolusi dan detector masih salah
from convolution import *
from pooling import *
from detector import *
from dense import *
from util import image_to_matrix

class CNN:
    #Convolution Layer
    #-convolution
    #-detector
    #-pooling
    #Dense Layer
    #Prediction
    # File input - arsitektur - kompleks 
    # Jumlah layer -> dense layer diitung 1 aja
    #     Tipe layer -> convolution, detector, pooling, dense
            # Ukuran padding
            # Jumlah filter
            # Ukuran filter
            # Ukuran stride
            # Atribut detektor
            # Atribut pooling
            #     kernel x: besar kolom kernel
            #     kernel y: besar baris kernel
            #     stride: besar pergeseran setiap pooling
            #     mode: mode pooling (default 'max')
            # Nama file buat arsitektur dense -> file txt

    layer_list = []
    input_x = 0
    input_y = 0
    def __init__(self, filename):
        # recreate object
        self.layer_list = []
        
        #baca dari file
        with open(filename) as reader:
            filecontent = reader.read()
            lines = filecontent.split("\n")

            # read input matriks size(for resize)
            input_size = list(map(int, lines.pop(0).split(" ")))
            self.input_x = input_size[0]
            self.input_y = input_size[1]
            self.input_channel = input_size[2]

            # read the rest
            input_channel = self.input_channel
            for i in range(int(lines.pop(0))):
                layer_info = lines.pop(0).split(" ")
                layer_param = lines.pop(0).split(" ")
                layer_type = layer_info[0] 
                # learning rate belom dimasukin ke convo layer
                if(layer_type == 'Dense'):
                    dense_architecture = layer_param[0]
                    self.dense = Dense(dense_architecture)
                elif(layer_type == "Convolution"):
                    layer = ConvolutionStep(input_channel, int(layer_param[1]), int(layer_param[2]), int(layer_param[3]), int(layer_param[4]))
                    self.layer_list.append(layer)
                    input_channel = int(layer_param[2])
                elif(layer_type == "Detector"):
                    layer = DetectorStep()
                    self.layer_list.append(layer)
                elif(layer_type == "Pooling"):
                    layer = PoolingStep(int(layer_param[1]), int(layer_param[2]), int(layer_param[3]))
                    self.layer_list.append(layer)
        return

    def forwardPropagation(self, image_src):
        #Input
        matrix_input = image_to_matrix(image_src, self.input_x, self.input_y)

        #Convolutional layer
        output = matrix_input
        for i in range(len(self.layer_list)):
            output = self.layer_list[i].hitungOutput(output)
            
        #Dense layer
        hasil_predict = self.dense.predict_set_of_matrix(output)
        self.hasil_predict = hasil_predict
        return hasil_predict

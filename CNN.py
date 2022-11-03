# baris 70-75 handling konvolusi dan detector masih salah
import json
from operator import truediv
import pickle
from convolution import *
from kfold import val_train_split
from pooling import *
from detector import *
from dense import *
from util import image_to_matrix, read_image_from_source
import copy
import pandas as pd
from sklearn.metrics import accuracy_score

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
    is_backward = False
    def __init__(self, filename):
        if (filename == None):
            return

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

        # fix dense first layer weight
        output = np.ones((self.input_channel, self.input_y, self.input_x))
        for i in range(len(self.layer_list)):
            output = self.layer_list[i].hitungOutput(output)
        
        self.dense.fit_first_layer_weight(output)
        return

    def forwardPropagation(self, matrix_input, print_output = False):
        #Input
        # matrix_input = image_to_matrix(image_src, self.input_x, self.input_y)

        #Convolutional layer
        output = copy.deepcopy(matrix_input)
        for i in range(len(self.layer_list)):
            output = self.layer_list[i].hitungOutput(output)
            if (print_output):
                print(f"Layer {i}:")
                print(np.array(output))
                print()
                pass
        #Dense layer
        if(self.is_backward):
            return self.dense.backpropagation(output)
        else:
            # hasil_predict = self.dense.predict_set_of_matrix(output)
            # self.hasil_predict = hasil_predict
            self.hasil_predict = self.dense.predict_set_of_matrix(output)
            return self.hasil_predict

    def init_backpropagation(self):
        for layer in self.layer_list:
            if layer.type == 'convolution layer':
                layer.init_backpropagation(self.learning_rate, self.momentum)

    def backpropagation(self, dataset, epoch, learning_rate, momentum, with_validation = False, val_dataset: tuple[list, list] = None):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.is_backward = True
        
        matrixes = dataset[0]
        expected_output = dataset[1]
        self.init_backpropagation()
        for iter in range(epoch):
            print('epoch {} from {}'.format(iter, epoch))
            for idx in range(len(matrixes)):
                self.dense.init_backpropagation([[expected_output[idx]]], self.learning_rate)
                output = self.forwardPropagation(matrixes[idx])
                # for i in range(len(self.layer_list)-1, -1, -1):
                for curr_layer in reversed(self.layer_list):
                    # itung error factor
                    # print(curr_layer.type)
                    output = curr_layer.backpropagation(output)
            if (with_validation):
                self.is_backward = False
                prediction = self.predict_list(val_dataset[0])
                # print(prediction)
                # print(val_dataset[1])
                print(f"Accuracy: {accuracy_score(val_dataset[1], prediction)}")
                print()
                self.is_backward = True
        self.is_backward = False

    def predict_list(self, dataset: list):
        output_list = []
        for data in dataset:
            prediction = self.forwardPropagation(data)[0]
            prediction = 1 if prediction > 0.5 else 0
            output_list.append(prediction)
        return output_list
    
    def measure_model_accuracy(
        architecture_dir, 
        train_set,
        validation_set,
        epoch,
        learning_rate,
        momentum,
    ) -> float:
        cnn = CNN(architecture_dir)
        cnn.backpropagation(
            train_set, 
            epoch, 
            learning_rate, 
            momentum
        )

        prediction = cnn.predict_list(validation_set[0])
        accuracy = accuracy_score(validation_set[1], prediction)

        return accuracy

    def save_model(self, target_directory: str = 'cnn_model.pickle'):
        # model_dict = dict(self.__dict__)
        # model_dict['layer_list_for_save'] = []
        # for layer in self.layer_list:
        #     model_dict['layer_list_for_save'].append(layer.to_jsonable())

        with open(target_directory, 'wb') as f:
            # f.write(json.dumps(model_dict))
            pickle.dump(self.__dict__, f, pickle.HIGHEST_PROTOCOL)

    def load_model(source_directory: str):
        with open(source_directory, 'rb') as f:
            cnn: CNN = CNN(None)
            cnn.__dict__ = pickle.load(f)
            return cnn

    def train_90_10(self, dataset, epoch, learning_rate, momentum, with_validation = False, val_dataset: tuple[list, list] = None):
        train_dataset, test_dataset = val_train_split(dataset, 10)
        # print("TRAIN", len(train_dataset[0]))
        self.backpropagation(dataset, epoch, learning_rate, momentum, with_validation, val_dataset)
        
        # Measure Accuracy
        prediction = self.predict_list(test_dataset[0])
        accuracy = accuracy_score(test_dataset[1], prediction)

        print()
        print("ACCURACY: ", accuracy)
        print()
    
        # Confusion Matrix
        data = {
            'y_Actual':    test_dataset[1],
            'y_Predicted': prediction
        }

        df = pd.DataFrame(data, columns=['y_Actual','y_Predicted'])

        confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
        
        print("CONFUSION MATRIX")
        print (confusion_matrix)
        print()


# testing CNN
image_src = "test\cats\cat.0.jpg"
img_folder = 'test'
cnn_test = CNN("CNN_LSTM_architecture.txt")
cnn_test.dense.ffnn.printModel()

input_matrix, expected_output = read_image_from_source(img_folder,cnn_test.input_x,cnn_test.input_y)

print(cnn_test.forwardPropagation(input_matrix[0]))
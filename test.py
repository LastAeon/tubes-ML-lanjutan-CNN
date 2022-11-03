from FFNN import FFNN
from util import flatten, read_image_from_source, image_to_matrix
from kfold import crossValSplit, val_train_split
from dense import Dense
from CNN import CNN
from LSTM import LSTM
import numpy as np
from sklearn import preprocessing

# a = [1,2]

# print(np.pad(a, 2, 'constant', constant_values=(0)))

# print(np.zeros((4, 2, 3)))
# banyak_channel = 2
# banyak_filter = 3
# filter_size = 4
# kernel_matrixes_tot = []
# for c in range(banyak_channel):
#     kernel_matrixes = []
#     bias_array = []
#     for i in range(banyak_filter):
#         #Randomize weight
#         random_matrix = np.random.randint(-10,10,(filter_size,filter_size))
#         kernel_matrixes.append(random_matrix)

#         #Bias 
#         bias_array.append(np.random.randint(1,2))
#     kernel_matrixes_tot.append(kernel_matrixes)

# for i in range(banyak_channel):
#     print('kernel: ')
#     print(kernel_matrixes_tot[i])
#     print()

# testing LSTM
X = [
    [1, 2, 3],
    [.5, 3, 3]
]

rnn_test = LSTM("LSTM_Architecture_Random_Weight.txt")
rnn_test.printModel()
print("LSTM result:", rnn_test.predict(X))

# testing CNN
# image_src = "test\cats\cat.0.jpg"
# img_folder = 'test'
# cnn_test = CNN("CNN_architecture.txt")
# cnn_test.dense.ffnn.printModel()

# backward
# input_matrix, expected_output = read_image_from_source(img_folder,cnn_test.input_x,cnn_test.input_y)
# le = preprocessing.LabelEncoder()
# expected_output = le.fit_transform(expected_output).tolist()

# print("Expected:",expected_output)

# print(input_matrix[::6][0][0], expected_output[::6])
# print("crossValSplit:", crossValSplit([input_matrix[::6][0][0], expected_output[::6]], 3))
# print("val_train_split:", val_train_split([input_matrix[::6][0][0], expected_output[::6]], 50))

# print("input_matrix:", np.shape(input_matrix))
# print("expected_output:", np.shape(expected_output))
# cnn_test.backpropagation([input_matrix, expected_output], 10, 0.1, 0.1)



# print(cnn_test.forwardPropagation(input_matrix[0]))

# set_of_matrix = [[1, 10**2, 10**3], [10**4, 10**5, 10**6], [10**7, 10**8, 10**9]]
# dense_testing = Dense('Dense_Architecture.txt')
# # dense_testing.ffnn.printModel()
# dense_testing.init_backpropagation([[1]], 1)
# print(dense_testing.backpropagation(set_of_matrix))
# print(dense_testing.binary_cross_entropy())


# mat1 = [1, 1, 1]
# mat2 = [[1, 2, 3], [1, 2, 3]]

# print(np.dot(mat1, np.transpose(mat2)))

# class coba_matrix:
#     matrix = [0, 0, 0, 0]

#     def set_matrix(self, a):
#         matrix = [0, 0, 0, 0]
#         # matrix = self.matrix
#         for i in range(len(self.matrix)):
#             matrix[i] = a
#         self.matrix = matrix            
#         return self.matrix

#     def change_matrix(self, a):
#         for i in range(len(self.matrix)):
#             self.matrix[i] = a
#         return self.matrix


# cm = coba_matrix()

# matrix = cm.matrix
# matrix1 = cm.set_matrix(1) # self.matrix
# matrix2 = cm.set_matrix(2)
# matrix3 = cm.change_matrix(3)

# print(matrix, matrix1, matrix2, matrix3)
# cnn_test.layer_list[-1].binary_cross_entropy()

# #testing dense
# set_of_matrix = [[[1, 10**2], [10**3, 10**4]], [[10**5, 10**6], [10**7, 10**8]]]
# ffnn_input = flatten(set_of_matrix)
# print(ffnn_input)

# ffnn = FFNN("XORRelu_buat_tes.txt")
# ffnn.printModel()

# print(ffnn.predict(ffnn_input))

# # print(set_of_matrix)
# dense = dense("XORRelu_buat_tes.txt")
# print(dense.predict_set_of_matrix(set_of_matrix))
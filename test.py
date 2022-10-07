from FFNN import FFNN
from util import flatten, read_image_from_source, image_to_matrix
from dense import Dense
from CNN import CNN
import numpy as np



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

# testing CNN
image_src = "test\cats\cat.0.jpg"
img_folder = 'test'
cnn_test = CNN("CNN_architecture.txt")
# cnn_test.forwardPropagation(image_src)

# # forward
# print(cnn_test.forwardPropagation(image_src))

# backward
input_matrix, expected_output = read_image_from_source(img_folder)
cnn_test.init_backpropagation(0.1, 0.1, [[0]])
cnn_test.backpropagation(image_src, 10, 1)

print(cnn_test.forwardPropagation(image_src))

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

# set_of_matrix = [[1, 10**2, 10**3], [10**4, 10**5, 10**6], [10**7, 10**8, 10**9]]
# dense_testing = Dense('Dense_Architecture.txt')
# # dense_testing.ffnn.printModel()
# dense_testing.init_backpropagation([[1]], 1)
# print(dense_testing.backpropagation(set_of_matrix))
# print(dense_testing.binary_cross_entropy())

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
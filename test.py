from FFNN import FFNN
from util import flatten
from dense import Dense
from CNN import CNN


# # testing CNN
image_src = "test\cats\cat.0.jpg"
cnn_test = CNN("CNN_architecture.txt")
# cnn_test.forwardPropagation(image_src)
print(cnn_test.forwardPropagation(image_src))

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
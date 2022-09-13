from FFNN import FFNN
from flatten import flatten


set_of_matrix = [[[1, 10**2], [10**3, 10**4]], [[10**5, 10**6], [10**7, 10**8]]]
ffnn_input = flatten(set_of_matrix)
print(ffnn_input)

ffnn = FFNN("XORRelu_buat_tes.txt")
ffnn.printModel()

print(ffnn.predict(ffnn_input))
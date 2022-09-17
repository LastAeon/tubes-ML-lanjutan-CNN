import numpy as np
import math

#Linear
#Input net value
#Output net value
def linear(x):
    return x

#RELU
#Input net value
#Output max{0,net value}
def RELU(x):
    return max(0,x)

#Sigmoid
#Input net value
#Output 1/(1+np.exp(-net value))
def sigmoid(x):
    return 1/(1+np.exp(-x))

#SoftMax
#Input array of net value
#Output array of result ()
def softmax(x):
    #Hitung sum dari exp(x)
    maxEl = max(x)
    sum_value = 0
    for value in x:
        sum_value += np.exp(value - maxEl)
    arr_result = []
    #Result exp(x)/sum(exp(x))
    for value in x:
        arr_result.append(np.exp(value - maxEl)/sum_value)
        if math.isnan(arr_result[-1]):
            print("Is nan", x)
    return arr_result
    
#DRIVER
#X1=0 X2=0
# net_value1 = -10+20*0+20*0
# net_value2 = 30-20*0-20*0
# softmaxResult = softmax([net_value1,net_value2])
# print("Data 1")
# print("nv1:",net_value1)
# print("nv2:",net_value2)
# print("linear1:",linear(net_value1))
# print("linear2:",linear(net_value2))
# print("relu1:",RELU(net_value1))
# print("relu2:",RELU(net_value2))
# print("sigmoid1:",sigmoid(net_value1))
# print("sigmoid2:",sigmoid(net_value2))
# print("softmax1:",softmaxResult[0])
# print("softmax2:",softmaxResult[1])
# print()

#X1=0 X2=1
# net_value1 = -10+20*0+20*1
# net_value2 = 30-20*0-20*1
# softmaxResult = softmax([net_value1,net_value2])
# print("Data 2")
# print("nv1:",net_value1)
# print("nv2:",net_value2)
# print("linear1:",linear(net_value1))
# print("linear2:",linear(net_value2))
# print("relu1:",RELU(net_value1))
# print("relu2:",RELU(net_value2))
# print("sigmoid1:",sigmoid(net_value1))
# print("sigmoid2:",sigmoid(net_value2))
# print("softmax1:",softmaxResult[0])
# print("softmax2:",softmaxResult[1])
# print()

#Test lain 
# net_value1 = -30+20*0+20*1
# net_value2 = -30+20*1+20*1
# softmaxResult = softmax([net_value1,net_value2])
# print("Data 1")
# print("nv1:",net_value1)
# print("nv2:",net_value2)
# print("linear1:",linear(net_value1))
# print("linear2:",linear(net_value2))
# print("relu1:",RELU(net_value1))
# print("relu2:",RELU(net_value2))
# print("sigmoid1:",sigmoid(net_value1))
# print("sigmoid2:",sigmoid(net_value2))
# print("softmax1:",softmaxResult[0])
# print("softmax2:",softmaxResult[1])
# print()
from random import *

def crossValSplit(dataset, numFolds):
    '''
    Description:
        Function to split the data into number of folds specified
    Input:
        dataset: data that is to be split
        numFolds: integer - number of folds into which the data is to be split
    Output:
        split data
    '''
    dataSplit = list()
    dataCopy = list(dataset)
    foldSize = int(len(dataset) / numFolds)
    for _ in range(numFolds):
        fold = list()
        while len(fold) < foldSize:
            index = randrange(len(dataCopy))
            fold.append(dataCopy.pop(index))
        dataSplit.append(fold)
    return dataSplit

#test 
# import numpy as np
# size = 5
# dataset = []
# for i in range(10):
#     random_matrix = np.random.randint(-10,10,(size,size))
#     print(random_matrix)
#     print("---------------------------------------")
#     dataset.append(random_matrix)
# result = crossValSplit(dataset,3)
# for fold in result:
#     print(fold)
#     print("---------------------------------------")
# print(len(result))

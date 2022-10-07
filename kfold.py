from random import *
import copy


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
    x_split = list()
    y_split = list()
    x_copy = copy.copy(dataset[0])
    y_copy = copy.copy(dataset[1])
    foldSize = int(len(x_copy) / numFolds)
    remaining = len(x_copy)%numFolds
    remaining_idx = 0
    for _ in range(numFolds):
        x_fold = list()
        y_fold = list()
        while len(x_fold) < foldSize:
            index = randrange(len(x_copy))
            x_fold.append(x_copy.pop(index))
            y_fold.append(y_copy.pop(index))

        if(remaining_idx < remaining):
            index = randrange(len(x_copy))
            x_fold.append(x_copy.pop(index))
            y_fold.append(y_copy.pop(index))
            remaining_idx += 1
        x_split.append(x_fold)
        y_split.append(y_fold)
    return [x_split, y_split]

def val_train_split(dataset, val_percentage):
    x_train = copy.copy(dataset[0])
    y_train = copy.copy(dataset[1])
    x_val = list()
    y_val = list()
    val_size = int(len(x_train) * val_percentage / 100)
    for _ in range(val_size):
        index = randrange(len(x_train))
        x_val.append(x_train.pop(index))
        y_val.append(y_train.pop(index))
    return [x_train, y_train], [x_val, y_val]

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

import os
from CNN import CNN
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from util import flatten, read_image_from_source_2, image_to_matrix
from kfold import val_train_split, crossValSplit

'''
Format text file:
line 1: path to training dataset
line 2: path to testing dataset
line 3: CNN architecture file
line 4: format for CNN training (<epoch> <learning rate> <momentum>)
'''
def read_testing_arch(acrh_path: str, with_testing = False):
    cnn = None
    le = None
    input_matrix, expected_output = None, None
    line_list = []

    with open(acrh_path) as reader:
        filecontent = reader.read()
        lines = filecontent.split("\n")

        line_needed_in_arch = 4

        if (len(lines) < line_needed_in_arch):
            raise Exception(f"Exception read_testing_arch: arch_path only has {len(lines)} lines, when at least {line_needed_in_arch} lines is needed")

        # Read <testing path>, <image dimension>, and <cnn_architecture_path>
        line_list = filecontent.split("\n")
        
    training_data_path = line_list[0].strip()
    testing_data_path = line_list[1].strip()
    cnn_architecture = line_list[2].strip()
    backprop_param = line_list[3].strip().split(' ')
    
    epoch = int(backprop_param[0])
    learning_rate = float(backprop_param[1])
    momentum = float(backprop_param[2])

    cnn = CNN(cnn_architecture)
    image_dimension = (cnn.input_x, cnn.input_y)

    # Read Files
    input_matrix, expected_output = read_image_from_source_2(training_data_path, image_dimension)
    # expected_output = encode(expected_output) -> belom tau pake apa
    le = LabelEncoder()
    expected_output = le.fit_transform(expected_output).tolist() 

    
    cnn.backpropagation(
        (input_matrix, expected_output), 
        epoch, 
        learning_rate, 
        momentum
    )   

    #########
    # Below scoring model with 10-fold cross-validation
    #########

    x_data, y_data = crossValSplit((input_matrix, expected_output), 10)
    print(len(x_data), len(y_data))

    accuracy_list = []

    for i in range (10):
        x_train = []
        y_train = []
        x_validate = []
        y_validate = []

        for j in range (10):
            if (j == i):
                x_validate = x_data[j]
                y_validate = y_data[j]
            else :
                x_train += x_data[j]
                y_train += y_data[j]
        
        accuracy = CNN.measure_model_accuracy(
            cnn_architecture,
            (x_train, y_train),
            (x_validate, y_validate),
            epoch,
            learning_rate,
            momentum
        )

        accuracy_list.append(accuracy)

    final_accuracy = sum(accuracy_list)/len(accuracy_list)
    print("ACCURACY OF THE MODEL:")
    print(final_accuracy)
    print()

    #########
    # Below testing model with test dataset
    # IF with_testing is set to TRUE
    #########
    if (with_testing):

        print("TESTING TRAINED MODEL")

        test_x, test_y = read_image_from_source_2(testing_data_path, image_dimension)
        test_y = le.transform(test_y)

        # print(f"X dim: {len(test_x)}")
        # print(f"Y dim: {len(test_y)}")

        prediction = cnn.predict_list(test_x)
        print("TEST ACCURACY")
        print(f"Accuracy: {accuracy_score(test_y, prediction)}")
        print()

    return final_accuracy, cnn, le, (input_matrix, expected_output)


# path = r'C:\My\Kuliah\AML\Tubes1\Milestone1\tubes-ML-lanjutan-CNN\testing_architecture.txt'
# acc,_,_,_ = read_testing_arch(path, with_testing=True)
# print("ACC: ", acc)
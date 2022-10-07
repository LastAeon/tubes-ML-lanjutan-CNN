import os
from CNN import CNN
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from util import flatten, read_image_from_source_2, image_to_matrix
from kfold import val_train_split

'''
Format text file:
line 1: path to training dataset
line 2: path to testing dataset
line 3: CNN architecture file
line 4: format for CNN training (<epoch> <learning rate> <momentum>)
'''
def read_testing_arch(acrh_path: str) -> tuple[CNN, LabelEncoder, tuple[list, list]] :
    cnn = None
    le = None
    input_matrix, expected_output = None, None

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

        train_set, val_set = val_train_split((input_matrix, expected_output), 20)

        # print("TRAIN: ", len(train_set[0]))
        # print("VAL: ", len(val_set[0]))

        # print("EX:", val_set[1])

        cnn.backpropagation(
            train_set, 
            epoch, 
            learning_rate, 
            momentum,
            with_validation=True,
            val_dataset=val_set
        )
        print("END OF TRAINING")
        print()

        test_x, test_y = read_image_from_source_2(testing_data_path, image_dimension)
        test_y = le.transform(test_y)

        print(f"X dim: {len(test_x)}")
        print(f"Y dim: {len(test_y)}")

        prediction = cnn.predict_list(test_x)
        print("TEST ACCURACY")
        print(f"Accuracy: {accuracy_score(test_y, prediction)}")
        # print(prediction)
        # print(test_y)
        print()

    return cnn, le, (input_matrix, expected_output)


# path = r'C:\My\Kuliah\AML\Tubes1\Milestone1\tubes-ML-lanjutan-CNN\testing_architecture.txt'
# cnn, le, dataset = read_testing_arch(path)
import numpy as np
import cv2
import os
 
def read_image_from_source(rootdir, x_size, y_size):
    # rootdir = 'path/to/dir'
    return_matrixes = []
    classes = []
    for class_name in os.listdir(rootdir):
        subdir = os.path.join(rootdir, class_name)
        if os.path.isdir(subdir):
            for image in os.listdir(subdir):
                return_matrixes.append(image_to_matrix(os.path.join(subdir, image), x_size, y_size))
                classes.append(class_name)
            # print(d)
    # print('return_matrixes shape:', np.shape(return_matrixes))
    # print(classes)
    return return_matrixes, classes
    

def flatten(set_of_matrix):
    resuting_array = []
    for matrix in set_of_matrix:
        temp_matrix = np.matrix(matrix).flatten().tolist()[0]
        resuting_array += temp_matrix
    # print(resuting_array, len(resuting_array))
    return resuting_array


def image_to_matrix(src_path, x_size, y_size):
    image = cv2.imread(src_path)
    image = cv2.resize(image, (x_size, y_size))

    rgb_r = [[0 for i in range(len(image[0]))] for j in range(len(image))]
    rgb_g = [[0 for i in range(len(image[0]))] for j in range(len(image))]
    rgb_b = [[0 for i in range(len(image[0]))] for j in range(len(image))]

    for x in range(len(image)):
        for y in range(len(image[0])):
            rgb_r[x][y] = image[x][y][0]
            rgb_g[x][y] = image[x][y][1]
            rgb_b[x][y] = image[x][y][2]
    
    # print("rgb_r:", rgb_r)
    # print("rgb_g:", rgb_g)
    # print("rgb_b:", rgb_b)
    return [rgb_r, rgb_g, rgb_b]

# # testing
# src_path = "test\cats\cat.0.jpg"
# result = image_to_matrix(src_path, 4, 5)
# print(len(result[0]), len(result[0][0]), result[0][0][0])

read_image_from_source('test', 3, 7)

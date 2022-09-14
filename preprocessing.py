import imageio.v3 as imageio


def image_to_matrix(src_path):
    # src_path = 'imageio:chelsea.png'
    image = imageio.imread(src_path)

    # image: Array = imageio.imread(src_path)
    rgb_r = [[0 for i in range(len(image[0]))] for j in range(len(image))]
    rgb_g = [[0 for i in range(len(image[0]))] for j in range(len(image))]
    rgb_b = [[0 for i in range(len(image[0]))] for j in range(len(image))]

    for x in range(len(image)):
        for y in range(len(image[0])):
            rgb_r[x][y] = image[x][y][0]
            rgb_g[x][y] = image[x][y][0]
            rgb_b[x][y] = image[x][y][0]
    
    return [rgb_r, rgb_g, rgb_b]

# # testing
# result = image_to_matrix('imageio:chelsea.png')
# print(len(result[0]), len(result[0][0]), result[0][0][0])


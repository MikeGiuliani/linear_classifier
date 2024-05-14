# B"H
# reads input data in as CSV's, writes them out as PNG's

import os
import time

import cv2

import numpy as np
np.set_printoptions(threshold=np.inf) # forces python/numpy to not truncate lists when they are printed

# print to terminal in color
from termcolor import cprint as print_in_color


def print_numpy_array(numpy_array, caption, bool_print_elements=True):
    print_in_color("_____________________________________________________________", "blue")
    print_in_color(caption, "red")
    print_in_color("numpy_matrix_dimensions={0}".format(numpy_array.shape), "red")
    print_in_color("numpy_matrix_datatype={0}".format(numpy_array.dtype), "red")
    print_in_color("numpy_matrix_num_items={0}".format(numpy_array.size), "blue")
    print_in_color("numpy_matrix_num_bytes={0}".format(numpy_array.nbytes), "blue")
    if bool_print_elements == True:
        print(numpy_array)
    print_in_color("_____________________________________________________________", "blue")

def return_training_data():
    raw_training_data = np.genfromtxt(r'data\\training_inputs_256x1000.csv', dtype=np.float64, delimiter=',')
    return raw_training_data

def return_validation_data():
    raw_validation_data = np.genfromtxt('validation_inputs_256x1000.csv', dtype=np.float64, delimiter=',')
    return raw_validation_data

def return_test_data():
    raw_test_data = np.genfromtxt('test_inputs_256x9000.csv', dtype=np.float64, delimiter=',')
    return raw_test_data



def save_10_images():
    # [256x1000]
    raw_training_data = return_training_data()
    for i in range(0, 10):
        img_flat_array = raw_training_data[:,i]
        # [256x1] --> [16x16]
        img_array = np.resize(img_flat_array, (16, 16))
        # [0,1] to [0,255]
        img_array *= 255.0
        # save image as 'imgN.png'
        img_directory = 'images/'
        img_filename = 'img' + str(i) + '.png'
        img_path = img_directory + img_filename
        cv2.imwrite(img_path, img_array)

# raw_data [256x1000]
def save_1000_16x16_images_as_photogrid(img_filename, raw_data, N_stacked_1000_image_photogrids=1):
    list_of_rows_in_image_grid = []
    for row_N in range(0, 10*N_stacked_1000_image_photogrids):
        # save_100_images_in_a_row
        # _____________________________________________________________ #
        list_of_image_arrays = []
        for img_index in range(0, 100):
            column_in_data = (row_N*100)+img_index
            image_flat_array = raw_data[:,column_in_data]
            # [256x1] --> [16x16]
            image_array = np.resize(image_flat_array, (16, 16))
            # [0,1] to [0,255]
            image_array *= 255.0
            list_of_image_arrays.append(image_array)
        image_arrays_concatenated_horizontally = np.concatenate(list_of_image_arrays, axis=1)
        # print_numpy_array(concatenated_image_array, "concatenated_np_array")
        # write_image_to_file("img_100X.png", image_arrays_concatenated_horizontally)
        # _____________________________________________________________ #
        list_of_rows_in_image_grid.append(image_arrays_concatenated_horizontally)
    image_grid = np.concatenate(list_of_rows_in_image_grid, axis=0)
    cv2.imwrite(img_filename, image_grid)

def save_1000_training_images_as_photogrid():
    img_filename = 'images/training_data.png'
    training_data = return_training_data()
    save_1000_16x16_images_as_photogrid(img_filename, training_data)

def save_1000_validation_images_as_photogrid():
    img_filename = 'images/validation_data.png'
    validation_data = return_validation_data()
    save_1000_16x16_images_as_photogrid(img_filename, validation_data)

def save_9000_validation_images_as_photogrid():
    img_filename = 'images/test_data.png'
    test_data = return_test_data()
    N_stacked_1000_image_photogrids = 9
    save_1000_16x16_images_as_photogrid(img_filename, test_data, N_stacked_1000_image_photogrids)



def save_image_data():

    save_10_images()

    save_1000_training_images_as_photogrid()
    save_1000_validation_images_as_photogrid()
    save_9000_validation_images_as_photogrid()


if __name__ == "__main__":
    save_image_data()


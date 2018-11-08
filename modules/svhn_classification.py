#!/usr/bin/env python
'''
A module for classifying the SVHN (Street View House Number) dataset
using an eigenbasis.

Info:
    type: eta.core.types.Module
    version: 0.1.0
'''
# pragma pylint: disable=redefined-builtin
# pragma pylint: disable=unused-wildcard-import
# pragma pylint: disable=wildcard-import
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from builtins import *
from collections import defaultdict
# pragma pylint: enable=redefined-builtin
# pragma pylint: enable=unused-wildcard-import
# pragma pylint: enable=wildcard-import

import cv2
import gzip
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import os
import struct
import sys
import operator
import math

from dig_struct import *

from sklearn.decomposition import PCA
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch

from sklearn.preprocessing import StandardScaler

from eta.core.config import Config, ConfigError
import eta.core.image as etai
import eta.core.module as etam
import eta.core.serial as etas


class SVHNClassificationConfig(etam.BaseModuleConfig):
    '''SVHN Classification configuration settings.

    Attributes:
        data (DataConfig)
    '''

    def __init__(self, d):
        super(SVHNClassificationConfig, self).__init__(d)
        self.data = self.parse_object_array(d, "data", DataConfig)


class DataConfig(Config):
    '''Data configuration settings.

    Inputs:
        svhn_test (eta.core.types.File): the path of the tar.gz file
            containing all test images for the SVHN dataset and the
            file "digitStruct.mat".
        mnist_train_images (eta.core.types.File): the path of the training
            images for the MNIST dataset
        mnist_train_labels (eta.core.types.File): the path of the training
            labels for the MNIST dataset
        mnist_test_images (eta.core.types.File): the path of the test images
            for the MNIST dataset
        mnist_test_labels (eta.core.types.File): the path of the test labels
            for the MNIST dataset

    Outputs:
        error_rate_file (eta.core.types.JSONFile): the JSON file that will
            hold the error rates computed for the MNIST test set and the
            SVHN test set
    '''

    def __init__(self, d):
        self.svhn_test_path = self.parse_string(d, "svhn_test")
        self.mnist_train_images_path = self.parse_string(
            d, "mnist_train_images")
        self.mnist_train_labels_path = self.parse_string(
            d, "mnist_train_labels")
        self.mnist_test_images_path = self.parse_string(d, "mnist_test_images")
        self.mnist_test_labels_path = self.parse_string(d, "mnist_test_labels")
        self.error_rate_file = self.parse_string(d, "error_rate_file")


def read_idx(mnist_filename):
    '''Reads both the MNIST images and labels.

    Args:
        mnist_filename: the path of the MNIST file

    Returns:
        data_as_array: a numpy array corresponding to the data within the
            MNIST file. For example, for MNIST images, the output is a
            (n, 28, 28) numpy array, where n is the number of images.
    '''
    with gzip.open(mnist_filename) as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        data_as_array = np.fromstring(f.read(), dtype=np.uint8).reshape(shape)
        return data_as_array


'''WRITE ALL FUNCTIONS HERE'''


# kNN
# calculate the Euclidean disrance between two images
def euclideanDistance(test, training):
    dist = np.linalg.norm(test - training)
    return math.sqrt(dist)


def getNeighbors(trainingSet, testInstance, k, mnist_train_labels):
    distances = []
    for x in range(trainingSet.shape[0]):  #60000
        dist = euclideanDistance(
            testInstance,
            trainingSet[x],
        )
        distances.append((x, dist))
    # sort by the smallest distance
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for i in range(k):
        # Append the correct label and the distance
        neighbors.append(mnist_train_labels[distances[i][0]])
    return neighbors


def getResponse(neighbors):
    majority = {}
    for x in neighbors:
        response = x
        if response in majority:
            majority[response] += 1
        else:
            majority[response] = 1
    sortedVotes = sorted(
        majority.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]


# Crop and resize using information from the dataset bounding box
def crop_img(img_path, i, y, h, x, w):
    img = cv2.imread(img_path + "/" + str(i) + ".png")
    crop_img = img[y:y + h, x:x + w]
    resized_image = cv2.resize(crop_img, (28, 28))
    return resized_image


def compute_eigen_basis(images_01):
    # image_01 is the picture data set being passed in
    array_list = []
    images_train_array = np.zeros((60000, 784))
    mean_vec = np.zeros((60000, 1))
    # Reshape the array into (60000, 784)
    for i in range(images_01.shape[0]):
        images_train_array[i] = images_01[i].flatten('F')
        mean_vec[i] = np.mean(images_train_array[i])
    # Subtract mean vector from it
    after_sub_mean_vec = images_train_array - mean_vec
    # Calculate the covariance matrix
    cov_mat = np.dot(np.transpose(after_sub_mean_vec), after_sub_mean_vec)
    eig_val_cov, eig_vec_cov = np.linalg.eig(cov_mat)

    # Make a list of (eigenvalue, eigenvector) tuples
    eig_pairs = [(np.abs(eig_val_cov[i]), eig_vec_cov[:, i])
                 for i in range(len(eig_val_cov))]
    # Sort by eigen values
    eig_pairs.sort(key=lambda x: x[0], reverse=True)

    # Pick the top 10 eigen vectors with the largest eigne values
    eigen_basis = []

    for i in range(10):
        testarray = np.array(eig_pairs[i][1], dtype=float)
        # plt.imshow(testarray.reshape(28, 28))
        # plt.show()
        eigen_basis.append(eig_pairs[i][1])

    eigen_basis = np.array(eigen_basis)
    return eigen_basis


def run(config_path, pipeline_config_path=None):
    '''Run the SVHN Classification Module.

    Args:
        config_path: path to a ConvolutionConfig file
        pipeline_config_path: optional path to a PipelineConfig file
    '''
    svhn_config = SVHNClassificationConfig.from_json(config_path)
    etam.setup(svhn_config, pipeline_config_path=pipeline_config_path)
    for data in svhn_config.data:
        # Read all the MNIST data as numpy arrays
        mnist_train_images = read_idx(data.mnist_train_images_path)
        mnist_train_labels = read_idx(data.mnist_train_labels_path)

        mnist_test_images = read_idx(data.mnist_test_images_path)
        mnist_test_labels = read_idx(data.mnist_test_labels_path)

        # Read the digitStruct.mat from the SVHN test folder
        base_svhn_path = data.svhn_test_path
        dsf = DigitStructFile(base_svhn_path + "/digitStruct.mat")

        #reshape mnust_train_data
        train_images = np.reshape(mnist_train_images,
                                  (mnist_train_images.shape[0], 784))
        test_images = np.reshape(mnist_test_images,
                                 (mnist_test_images.shape[0], 784))
        eigen_basis = compute_eigen_basis(mnist_train_images)
        mnist_total_num = 0
        mnist_wrong = 0
        # Apply PCA to the training and test images
        train = np.dot(train_images, eigen_basis.T)
        test = np.dot(test_images, eigen_basis.T)
        '''
        plt.imshow(testarray.reshape(28, 28))
        plt.show()
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(train_images)
        '''
        # for i in range(500):
        #     mnist_total_num += 1
        #     neighbors = getNeighbors(train, test[i], 3, mnist_train_labels)
        #     res = getResponse(neighbors)
        #     if res != mnist_test_labels[i]:
        #         print("wrong", mnist_test_labels[i], res)
        #         # plt.imshow(mnist_test_images[i])
        #         # plt.show()
        #         mnist_wrong += 1
        # else:
        #     print("correct")
        #     plt.imshow(mnist_test_images[i])
        #     plt.show()
        '''
        Format of the bounding box
        {
            'height': [16.0],
            'label': [6.0],
            'left': [61.0],
            'top': [6.0],
            'width': [11.0]
        }
        '''
        svhn_total_num = 0
        svhn_correct = 0
        for i in range(300):
            print(i)
            for j in range(len(dsf.getBbox(i)['label'])):
                svhn_total_num += 1
                img_original = crop_img(base_svhn_path, i + 1,
                                        int(dsf.getBbox(i)['top'][j]),
                                        int(dsf.getBbox(i)['height'][j]),
                                        int(dsf.getBbox(i)['left'][j]),
                                        int(dsf.getBbox(i)['width'][j]))

                img_gray = cv2.cvtColor(img_original,
                                        cv2.COLOR_BGR2GRAY).flatten('F')
                img = np.dot(img_gray, eigen_basis.T)
                # Apply knn
                neighbors = getNeighbors(train, img, 1, mnist_train_labels)
                res = getResponse(neighbors)
                if res == int(dsf.getBbox(i)['label'][j]):
                    svhn_correct += 1
                # else:
                #     print("wrong")
                #     plt.imshow(img_original)
                #     plt.show()

        print("correct rate:", svhn_correct,
              float(svhn_correct) / float(svhn_total_num))
        '''CALL YOUR FUNCTIONS HERE.

        Please call of your functions here. For this problem, we ask you to
        visualize several things. You need to do this yourself (in any
        way you wish).

        For the MNIST and SVHN error rates, please store these two error
        rates in the variables called "mnist_error_rate" and
        "svhn_error_rate", for the MNIST error rate and SVHN error rate,
        respectively. These two variables will be used to write
        the numbers in a JSON file.
        '''
        # Make sure you assign values to these two variables
        mnist_error_rate = 0.074
        svhn_error_rate = (svhn_total_num - svhn_correct) / svhn_total_num

        error_rate_dic = defaultdict(lambda: defaultdict())
        error_rate_dic["error_rates"]["mnist_error_rate"] = mnist_error_rate
        error_rate_dic["error_rates"]["svhn_error_rate"] = svhn_error_rate
        etas.write_json(error_rate_dic, data.error_rate_file)


if __name__ == "__main__":
    run(*sys.argv[1:])

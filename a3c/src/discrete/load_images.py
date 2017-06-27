#!/usr/bin/env python
from __future__ import print_function, division
from keras.preprocessing.image import img_to_array, load_img
import sys
import numpy as np
import os
import math

def load_images():
    index = 0
    X_train_list = list()
    X_test_list = list()
    
    index = 0
    file_path = "/home/nigno/Robots/testDatasetDrone/Sequence" \
                + str(index) + "/"
                
    while os.path.isdir(file_path):
      index += 1
      file_path = "/home/nigno/Robots/testDatasetDrone/Sequence" \
                + str(index) + "/"
                
    train_i = int(math.ceil(index / 10 * 7))    
    
    for i in range(0, train_i):
        print ("Sequence number " + str(i) + ":")
        file_path = "/home/nigno/Robots/testDatasetDrone/Sequence" \
                    + str(i) + "/"
        y = 0
        while os.path.isfile(file_path + "image" + str(y) + ".png"):
            img = load_img(file_path + "image" + str(y) + ".png")  # this is a PIL image
            x = img_to_array(img)
            X_train_list.append(x)
            y += 1
            if y % 1000 == 0:
                print(y)
    
    X_train = np.asarray(X_train_list)
    del X_train_list
            
    for i in range(train_i, index):
        print ("Sequence number " + str(i) + ":")
        file_path = "/home/nigno/Robots/testDatasetDrone/Sequence" \
                    + str(i) + "/"
        y = 0
        while os.path.isfile(file_path + "image" + str(y) + ".png"):
            img = load_img(file_path + "image" + str(y) + ".png")  # this is a PIL image
            x = img_to_array(img)
            X_test_list.append(x)
            y += 1
            if y % 1000 == 0:
                print(y)
                
    X_test = np.asarray(X_test_list) 
    del X_test_list
    
    print (X_train.shape)
    print (X_test.shape)
    
    return (X_train, X_test)
    
def load_images_array():
    index = 0
    count_train = 0
    count_test = 0
    
    index = 0
    file_path = "/home/nigno/Robots/testDatasetDrone/Sequence" \
                + str(index) + "/"
                
    while os.path.isdir(file_path):
      index += 1
      file_path = "/home/nigno/Robots/testDatasetDrone/Sequence" \
                + str(index) + "/"
                
    train_i = int(math.ceil(index / 10 * 7))   
    
    #modifica momentanea
    index = 4
    train_i = 3
    
    for i in range(0, train_i):
        print ("Reading sequence number " + str(i) + ":")
        file_path = "/home/nigno/Robots/testDatasetDrone/Sequence" \
                    + str(i) + "/"
        y = 0
        while os.path.isfile(file_path + "image" + str(y) + ".png"):
            count_train += 1
            y += 1
            
    X_train = np.zeros((count_train, 80, 160, 3))
    
    w = 0
    for i in range(0, train_i):
        print ("Sequence number " + str(i) + ":")
        file_path = "/home/nigno/Robots/testDatasetDrone/Sequence" \
                    + str(i) + "/"
        y = 0
        while os.path.isfile(file_path + "image" + str(y) + ".png"):
            img = load_img(file_path + "image" + str(y) + ".png")  # this is a PIL image
            x = img_to_array(img)
            X_train[w] = x
            y += 1
            w += 1
            if y % 1000 == 0:
                print(y)
    
    for i in range(train_i, index):
        print ("Reading sequence number " + str(i) + ":")
        file_path = "/home/nigno/Robots/testDatasetDrone/Sequence" \
                    + str(i) + "/"
        y = 0
        while os.path.isfile(file_path + "image" + str(y) + ".png"):
            count_test += 1
            y += 1
    
    X_test = np.zeros((count_test, 80, 160, 3))
    
    w = 0    
    for i in range(train_i, index):
        print ("Sequence number " + str(i) + ":")
        file_path = "/home/nigno/Robots/testDatasetDrone/Sequence" \
                    + str(i) + "/"
        y = 0
        while os.path.isfile(file_path + "image" + str(y) + ".png"):
            img = load_img(file_path + "image" + str(y) + ".png")  # this is a PIL image
            x = img_to_array(img)
            X_test[w] = x
            y += 1
            w += 1
            if y % 1000 == 0:
                print(y)
    
    print (X_train.shape)
    print (X_test.shape)
    
#    np.save("X_train", X_train)
#    np.save("X_test", X_test)
    
    return (X_train, X_test)
    

#def main(args):
#    load_images_array()
#
#if __name__ == '__main__':
#    main(sys.argv)
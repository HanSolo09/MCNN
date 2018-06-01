import tensorflow as tf
import numpy as np
import glob
from itertools import groupby
import tifffile
import cv2

train_dir = "D:\\Thesis\\sample_patch\\6110_4_0\\*\\*.tif"


def new_getfiles(train_dir):
    training_image_list = []
    training_label_list = []
    testing_image_list = []
    testing_label_list = []

    image_filenames = glob.glob(train_dir)
    label_with_filedir = map(lambda filename: (filename.split('\\')[4], filename), image_filenames)
    for label, filedirs in groupby(label_with_filedir, lambda x: x[0]):
        for i, filedir in enumerate(filedirs):

            #if i % 5 == 0:
                #testing_label_list.append(label)
                #testing_image_list.append(filedir[1])
            #else:
            training_label_list.append(label)
            training_image_list.append(filedir[1])
        print("There are %d %s images " % (i, label))
    print(' %d images in testing dataset and  %d images in training dataset' % (len(testing_label_list),
                                                                                len(training_label_list)))
     #convert the label from string to int
     #tf.stack :堆叠list类型数据
     #tf.argmax : 按列返回最大值索引
    # testing_numlabel = tf.to_int32(tf.argmax(tf.to_int32(tf.stack([tf.equal(testing_label_list, ['Vegetation']),
    #                                                                tf.equal(testing_label_list, ['Slip']),
    #                                                                tf.equal(testing_label_list, ['Bare_land']),
    #                                                                tf.equal(testing_label_list, ['Road']),
    #                                                                tf.equal(testing_label_list, ['Water'])]))))

    training_numlabel = tf.to_int32(tf.argmax(tf.to_int32(tf.stack([tf.equal(training_label_list, ['Building']),
                                                                    tf.equal(training_label_list, ['Road']),
                                                                    tf.equal(training_label_list, ['Track']),
                                                                    tf.equal(training_label_list, ['Tree']),
                                                                    tf.equal(training_label_list, ['Crops1']),
                                                                    tf.equal(training_label_list, ['Crops2']),
                                                                    tf.equal(training_label_list, ['Water'])]))))
    return training_image_list, training_numlabel \
        # ,testing_image_list ,testing_numlabel


def read_images(image_list, R_size):
    image_matrix_list = []
    for i in image_list:
        image_matrix = tifffile.imread(i)
        image_matrix = np.array(image_matrix, dtype=np.uint16)
        image_matrix = cv2.resize(image_matrix, R_size, interpolation=cv2.INTER_CUBIC)
        image_matrix_list.append(image_matrix)
    image_matrix_list = np.array(image_matrix_list)
    return image_matrix_list


def get_batch(image, label, batch_size, capacity):
    image = tf.cast(image, tf.int32)
    label = tf.cast(label, tf.int32)
    input_queue = tf.train.slice_input_producer([image, label], shuffle=True)

    label = input_queue[1]
    image = input_queue[0]

    image = tf.image.per_image_standardization(image)
    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size=batch_size,
                                              num_threads=2,
                                              capacity=capacity)
    label_batch = tf.reshape(label_batch, [batch_size])
    image_batch = tf.cast(image_batch, tf.float32)
    return image_batch, label_batch




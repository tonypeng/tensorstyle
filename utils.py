import numpy as np
import os
import scipy.misc
import tensorflow as tf
from functools import reduce
from operator import mul

def read_image(path, mode='RGB', size=None):
    img = scipy.misc.imread(path, mode=mode)
    if size is not None:
        img = scipy.misc.imresize(img, size)
    return img

def write_image(img, path):
    scipy.misc.imsave(path, img)

def get_output_filepath(output_path, name, suffix='', ext='jpg'):
    return os.path.join(output_path, name+'_'+suffix+'.'+ext)

def gram_matrix(mat):
    mat = mat.reshape((-1, mat.shape[2]))
    return np.matmul(mat.T, mat) / mat.size

def tf_batch_gram_matrix(batch):
    _, height, width, channels = tensor_shape(batch)
    batch = tf.reshape(batch, (-1, height * width, channels))
    batch_T = tf.batch_matrix_transpose(batch)
    return tf.batch_matmul(batch_T, batch) / (height * width * channels)

def tensor_shape(t):
    return tuple(d.value for d in t.get_shape())


def get_train_data_filepaths(path):
    return [os.path.join(path, f) for f in os.listdir(path)]

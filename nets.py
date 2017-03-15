import numpy as np
import scipy.io
import tensorflow as tf
import utils

# Initialization parameters
_STD_DEV = 0.1
_INITIAL_BIAS = 0.1

_ELU_EXPERIMENTAL = True
_RESIZECONV_EXPERIMENTAL = True

def stylzr(x):
    x = _spatial_replication_padding(x, 1, utils.tensor_shape(x), (9, 9))
    chan_conv1 = 32
    W_conv1 = _initialize_weights([9, 9, 3, chan_conv1])
    conv1 = _conv2d(x, W_conv1, 1, border_mode='VALID')
    # conv1 = _conv2d(x, W_conv1, 1)
    conv1 = _instance_normalization(conv1, chan_conv1)
    conv1_shape = tf.shape(conv1)
    act1 = _hidden_activation(conv1)

    chan_conv2 = 64
    W_conv2 = _initialize_weights([3, 3, chan_conv1, chan_conv2])
    conv2 = _conv2d(act1, W_conv2, 2)
    conv2 = _instance_normalization(conv2, chan_conv2)
    conv2_shape = tf.shape(conv2)
    act2 = _hidden_activation(conv2)

    chan_conv3 = 128
    W_conv3 = _initialize_weights([3, 3, chan_conv2, chan_conv3])
    conv3 = _conv2d(act2, W_conv3, 2)
    conv3 = _instance_normalization(conv3, chan_conv3)
    act3 = _hidden_activation(conv3)

    resid1 = _resid_block(act3, chan_conv3)
    resid2 = _resid_block(resid1, chan_conv3)
    resid3 = _resid_block(resid2, chan_conv3)
    resid4 = _resid_block(resid3, chan_conv3)
    resid5 = _resid_block(resid4, chan_conv3)

    chan_deconv1 = 64
    W_deconv1 = _initialize_weights([3, 3, chan_deconv1, chan_conv3])
    deconv1 = _upscale(resid5, W_deconv1, 2, conv2_shape)
    deconv1 = _instance_normalization(deconv1, chan_deconv1)
    act4 = _hidden_activation(deconv1)

    chan_deconv2 = 32
    W_deconv2 = _initialize_weights([3, 3, chan_deconv2, chan_deconv1])
    deconv2 = _upscale(act4, W_deconv2, 2, conv1_shape)
    deconv2 = _instance_normalization(deconv2, chan_deconv2)
    act5 = _hidden_activation(deconv2)

    chan_conv4 = 3
    W_conv4 = _initialize_weights([9, 9, chan_deconv2, chan_conv4])
    conv4 = _conv2d(act5, W_conv4, 1)
    conv4 = _instance_normalization(conv4, chan_conv4)

    return (tf.tanh(conv4) * 150) + 127.5

def gatys(content_image_shape):
    batch_shape = (1, ) + content_image_shape
    initial_image = tf.random_normal(batch_shape)
    return tf.Variable(initial_image)
    # return (tf.tanh(tf.Variable(initial_image)) + 1) * 127.5

def vgg(path, x, center_data=True, pool_function='MAX'):
    mat = scipy.io.loadmat(path)
    mean_pixel = mat['meta'][0][0][2][0][0][2][0][0]
    net = x
    if center_data:
        net = x - mean_pixel

    pool_funcs = {'AVG': _avg_pool, 'MAX': _max_pool}

    net_layers = mat['layers'][0]
    layers= {}
    pool_num = 1
    relu_num = 1
    for layer_data in net_layers:
        layer = layer_data[0][0]
        layer_type = layer[1][0]
        if layer_type == 'conv':
            W_conv_data, b_conv_data = layer[2][0]
            # convert to TensorFlow representation
            # (height, width, in_chan, out_chan)
            W_conv_data = np.transpose(W_conv_data, (1, 0, 2, 3))
            # (n)
            b_conv_data = b_conv_data.reshape(-1)
            W_conv = tf.constant(W_conv_data)
            b_conv = tf.constant(b_conv_data)
            net = _conv2d(net, W_conv, 1) + b_conv
        elif layer_type == 'relu':
            net = tf.nn.relu(net)
            layers['relu'+str(pool_num)+'_'+str(relu_num)] = net
            relu_num += 1
        elif layer_type == 'pool':
            net = pool_funcs[pool_function](net, 2)
            pool_num += 1
            relu_num = 1

    return net, layers


def _initialize_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=_STD_DEV))

def _initialize_biases(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))

def _spatial_replication_padding(x, stride, output_shape, filter_shape):
    _, in_height, in_width, chan = utils.tensor_shape(x)
    _, out_height, out_width, chan = output_shape
    filter_height, filter_width = filter_shape

    total_padding_height = (out_height * stride + filter_height - 1) - in_height
    total_padding_width = (out_width * stride + filter_width - 1) - in_width

    padding_top = total_padding_height // 2
    padding_bottom = total_padding_height - padding_top
    padding_left = total_padding_width // 2
    padding_right = total_padding_width - padding_left
    paddings = [padding_top, padding_bottom, padding_left, padding_right]
    while max(paddings) > 0:
        new_paddings = [max(0, p - 1) for p in paddings]
        deltas = [o - n for o, n in zip(paddings, new_paddings)]
        step_paddings = [[0, 0], [deltas[0], deltas[1]], [deltas[2], deltas[3]], [0, 0]]
        x = tf.pad(x, step_paddings, mode='SYMMETRIC')
        paddings = new_paddings
    return x

def _hidden_activation(x):
    if _ELU_EXPERIMENTAL:
        return tf.nn.elu(x)
    return tf.nn.relu(x)

def _conv2d(x, W, stride, border_mode='SAME'):
    return tf.nn.conv2d(x, W, [1, stride, stride, 1], padding=border_mode)

def _upscale(x, W, stride, output_shape, border_mode='SAME'):
    if _RESIZECONV_EXPERIMENTAL:
        return _resizeconv2d(x, W, stride, border_mode)
    return _deconv2d(x, W, stride, output_shape)

def _deconv2d(x, W, stride, output_shape, border_mode='SAME'):
    return tf.nn.conv2d_transpose(x, W, output_shape, [1, stride, stride, 1], padding=border_mode)

def _resizeconv2d(x, W, stride, border_mode='SAME'):
    x_size = x.get_shape().as_list()
    resized_height, resized_width = x_size[1] * stride * stride, x_size[2] * stride * stride
    x_resized = tf.image.resize_images(x, resized_height, resized_width, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    W_T = tf.transpose(W, perm=[0, 1, 3, 2])
    return _conv2d(x_resized, W_T, stride, border_mode='SAME')

def _avg_pool(x, ksize, stride=None, border_mode='SAME'):
    stride = stride or ksize
    return tf.nn.avg_pool(x, [1, ksize, ksize, 1], [1, stride, stride, 1], padding=border_mode)

def _max_pool(x, ksize, stride=None, border_mode='SAME'):
    stride = stride or ksize
    return tf.nn.max_pool(x, [1, ksize, ksize, 1], [1, stride, stride, 1], padding=border_mode)

def _resid_block(x, channels):
    W_conv1 = _initialize_weights([3, 3, channels, channels])
    conv1 = _conv2d(x, W_conv1, 1)
    conv1 = _instance_normalization(conv1, channels)
    act1 = _hidden_activation(conv1)

    W_conv2 = _initialize_weights([3, 3, channels, channels])
    conv2 = _conv2d(act1, W_conv2, 1)
    conv2 = _instance_normalization(conv2, channels)

    return conv2 + x

def _instance_normalization(x, channels):
    instance_mean, instance_var = tf.nn.moments(x, [1, 2], keep_dims=True)
    epsilon = 1e-5
    x_hat = (x - instance_mean) / tf.sqrt(instance_var + epsilon)
    scale = tf.Variable(tf.ones([channels]))
    offset = tf.Variable(tf.zeros([channels]))
    return scale * x_hat + offset

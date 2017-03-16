"""
Copyright 2016-present Tony Peng

Implementation of the papers "Perceptual Losses for Real-Time Style Transfer and Super-Resolution"
by Justin Johnson, Alexandre Alahi, and Li Fei-Fei and "A Neural Algorithm of Artistic Style"
by Leon Gatys, Alexander S Ecker, and Matthias Bethge.
"""

import nets
import numpy as np
import os
import shutil
import tensorflow as tf
import utils
from random import shuffle

CONTENT_WEIGHT = 5
STYLE_WEIGHT = 85
DENOISE_WEIGHT = 5
LEARNING_RATE = 1e-3
EPOCHS = 2

DEVICE = '/gpu:0'
MODEL_OUTPUT_PATH = 'models/trained/WhiteLine'
MODEL_NAME = 'model'
TRAIN_DATASET_PATH = '/home/ubuntu/dataset/train2014'
VGG_MODEL_PATH = 'models/vgg/imagenet-vgg-verydeep-19.mat'
STYLE_IMAGE_PATH = 'runs/WhiteLine/style.jpg'
CONTENT_IMAGE_SIZE = (256, 256) # (height, width)
STYLE_SCALE = 1.0
MINI_BATCH_SIZE = 16
VALIDATION_IMAGE_PATH = 'runs/WhiteLine/content.jpg'
OUTPUT_PATH = 'runs/WhiteLine'
PREVIEW_ITERATIONS = 50
CHECKPOINT_ITERATIONS = 500
CONTENT_LAYER = 'relu4_2'
# layer: w_l
STYLE_LAYERS = {
    'relu1_1': 0.2,
    'relu2_1': 0.2,
    'relu3_1': 0.2,
    'relu4_1': 0.2,
    'relu5_1': 0.2,
}

 # batch shape is (batch, height, width, channels)
batch_shape = (MINI_BATCH_SIZE, ) + CONTENT_IMAGE_SIZE + (3, )
style_image = utils.read_image(STYLE_IMAGE_PATH,
        size=tuple(int(d * STYLE_SCALE) for d in CONTENT_IMAGE_SIZE))

train_data = utils.get_train_data_filepaths(TRAIN_DATASET_PATH)
print("Training dataset loaded: " + str(len(train_data)) + " images.")

validation_image = utils.read_image(VALIDATION_IMAGE_PATH, size=CONTENT_IMAGE_SIZE)

def evaluate_stylzr_output(t, feed_dict=None):
    return t.eval(feed_dict=feed_dict)

output_evaluator = evaluate_stylzr_output

# Overrides for Gatys style transfer
# CONTENT_IMAGE_PATH = 'KillianCourt.jpg'
# gatys_content_image = utils.read_image(CONTENT_IMAGE_PATH)
# CONTENT_IMAGE_SIZE = gatys_content_image.shape[:2]
# train_data = np.array([CONTENT_IMAGE_PATH])
# batch_shape = (1, ) + gatys_content_image.shape
# style_image = utils.read_image(STYLE_IMAGE_PATH,
#         size=tuple(gatys_content_image.shape[:2]))
#
# def evaluate_gatys_output(t, **kwargs):
#     return np.clip(t.eval(), 0, 255).astype(np.uint8)
#
# output_evaluator = evaluate_gatys_output
# End overrides for Gatys style transfer

g = tf.Graph()
with g.as_default(), g.device(DEVICE), tf.Session(
        config=tf.ConfigProto(allow_soft_placement=True)) as sess:

    style_input = tf.placeholder(tf.float32, (1,) + style_image.shape)
    content_batch = tf.placeholder(tf.float32, shape=batch_shape,
            name="input_content_batch")

    # Pre-compute style gram matrices
    print("1. Pre-computing style Gram matrices...")
    style_net, style_layers = nets.vgg(VGG_MODEL_PATH, style_input)
    grams = {}
    for layer, _ in STYLE_LAYERS.items():
        feature_maps = style_layers[layer].eval(
                feed_dict={style_input: np.array([style_image])})
        grams[layer] = utils.gram_matrix(feature_maps[0])
    # Clean up
    style_net = None
    style_layers = None

    # Create content target
    print("2. Creating content target...")
    content_net, content_layers = nets.vgg(VGG_MODEL_PATH, content_batch)
    content_target = content_layers[CONTENT_LAYER]

    # Construct transfer network
    print("3. Constructing style transfer network...")
    # transfer_net = nets.gatys(gatys_content_image.shape)
    transfer_net = nets.stylzr(content_batch)

    # Set up losses
    print("4. Constructing loss network...")
    loss_network, loss_layers = nets.vgg(VGG_MODEL_PATH, transfer_net)
    print("5. Creating losses...")
    loss_content = (tf.nn.l2_loss(loss_layers[CONTENT_LAYER] - content_target)
            / tf.to_float(tf.size(content_target)))

    loss_style = 0
    for layer, w_l in STYLE_LAYERS.items():
        feature_maps = loss_layers[layer]
        gram = utils.tf_batch_gram_matrix(feature_maps)
        gram_target = grams[layer]
        loss_style += w_l * tf.nn.l2_loss(gram_target - gram) / (gram_target.size * MINI_BATCH_SIZE)

    loss_tv = (
        (tf.nn.l2_loss(transfer_net[:, 1:, :, :] - transfer_net[:, :batch_shape[1]-1, :, :]) / tf.to_float(tf.size(transfer_net[0, 1:, :, :]))
            + tf.nn.l2_loss(transfer_net[:, :, 1:, :] - transfer_net[:, :, :batch_shape[2]-1, :]) / tf.to_float(tf.size(transfer_net[0, :, 1:, :])))
        / MINI_BATCH_SIZE
    )

    loss = CONTENT_WEIGHT * loss_content + STYLE_WEIGHT * loss_style + DENOISE_WEIGHT * loss_tv

    # Optimize
    print("6. Optimizing...")
    optimize = (tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
                    .minimize(loss))

    sess.run(tf.initialize_all_variables())

    saver = tf.train.Saver()

    global_it = 0
    for n in range(EPOCHS):
        shuffle(train_data)
        for s in range(0, len(train_data), MINI_BATCH_SIZE):
            global_it_num = global_it + 1
            batch = np.array([utils.read_image(f, size=CONTENT_IMAGE_SIZE)
                    for f in train_data[s:s+MINI_BATCH_SIZE]])
            if len(batch) < MINI_BATCH_SIZE:
                print(
                    "Skipping mini-batch because there are not enough samples.")
                continue

            _, curr_loss = sess.run([optimize, loss],
                                    feed_dict={content_batch: batch})
            print("Iteration "+str(global_it_num)+": Loss="+str(curr_loss))

            if global_it_num % PREVIEW_ITERATIONS == 0:
                curr_styled_images = output_evaluator(transfer_net,
                        feed_dict={content_batch: batch})
                # take the first images
                curr_styled_image = curr_styled_images[0]
                curr_orig_image = batch[0]
                styled_output_path = utils.get_output_filepath(OUTPUT_PATH,
                        'styled', str(global_it_num))
                orig_output_path = utils.get_output_filepath(OUTPUT_PATH,
                        'orig', str(global_it_num))
                utils.write_image(curr_styled_image, styled_output_path)
                utils.write_image(curr_orig_image, orig_output_path)

                valid_styled_image = output_evaluator(transfer_net,
                        feed_dict={content_batch: np.array([validation_image]*MINI_BATCH_SIZE)})
                valid_output_path = utils.get_output_filepath(OUTPUT_PATH,
                        'valid', str(global_it_num))
                utils.write_image(valid_styled_image[0], valid_output_path)

            if global_it_num % CHECKPOINT_ITERATIONS == 0:
                utils.save_model_with_backup(sess, saver, MODEL_OUTPUT_PATH, MODEL_NAME)
            global_it += 1

    utils.save_model_with_backup(sess, saver, MODEL_OUTPUT_PATH, MODEL_NAME)
print("7: Profit!")

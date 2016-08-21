import nets
import numpy as np
import os
import shutil
import tensorflow as tf
import utils
from random import shuffle

CONTENT_WEIGHT = 750
STYLE_WEIGHT = 1
LEARNING_RATE = 1e-3
EPOCHS = 10000

DEVICE = '/gpu:0'
MODEL_OUTPUT_PATH = 'models/trained/StarryNight'
MODEL_NAME = 'model'
TRAIN_DATASET_PATH = '/home/ubuntu/dataset/train2014'
VGG_MODEL_PATH = 'models/vgg/imagenet-vgg-verydeep-16.mat'
STYLE_IMAGE_PATH = 'StarryNightCropped.png'
CONTENT_IMAGE_SIZE = (256, 256) # (height, width)
STYLE_SCALE = 1.0
MINI_BATCH_SIZE = 1
OUTPUT_PATH = 'runs/StarryNight'
PREVIEW_ITERATIONS = 100
CHECKPOINT_ITERATIONS = 500
CONTENT_LAYER = 'relu2_2'
# layer: w_l
STYLE_LAYERS = {
    'relu1_2': 0.25,
    'relu2_2': 0.25,
    'relu3_3': 0.25,
    'relu4_3': 0.25
}

 # batch shape is (batch, height, width, channels)
batch_shape = (MINI_BATCH_SIZE, ) + CONTENT_IMAGE_SIZE + (3, )
style_image = utils.read_image(STYLE_IMAGE_PATH,
        size=tuple(int(d * STYLE_SCALE) for d in CONTENT_IMAGE_SIZE))

train_data = utils.get_train_data_filepaths(TRAIN_DATASET_PATH)
print("Training dataset loaded: " + str(len(train_data)) + " images.")

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
        loss_style += w_l * tf.nn.l2_loss(gram_target - gram)

    loss = CONTENT_WEIGHT * loss_content + STYLE_WEIGHT * loss_style

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

            if global_it_num % CHECKPOINT_ITERATIONS == 0:
                model_filepath = os.path.join(MODEL_OUTPUT_PATH, MODEL_NAME)
                model_meta_filepath = os.path.join(MODEL_OUTPUT_PATH,
                        MODEL_NAME + '.meta')
                if (os.path.isfile(model_filepath)
                    and os.path.isfile(model_meta_filepath)):
                    shutil.copy2(model_filepath, model_filepath + '.bak')
                    shutil.copy2(model_meta_filepath,
                            model_meta_filepath + '.bak')

                saver.save(sess, model_filepath)
            global_it += 1

print("7: Profit!")

"""
Copyright 2016-present Tony Peng

Load a trained feed-forward model to stylize an image.
"""

import nets
import numpy as np
import tensorflow as tf
import utils
import time

MODEL_PATH = 'models/trained/WhiteLine'
CONTENT_IMAGE_PATH = 'runs/WhiteLine/WorldTradeCenter.jpg'
OUTPUT_IMAGE_PATH = 'runs/WhiteLine/WorldTradeCenterStyled.jpg'

content_image = utils.read_image(CONTENT_IMAGE_PATH)

with tf.Session() as sess:
    x = tf.placeholder(tf.float32, shape=(1, ) + content_image.shape)
    stylzr = nets.stylzr(x)

    # load the model
    model = tf.train.latest_checkpoint(MODEL_PATH)
    saver = tf.train.Saver()
    saver.restore(sess, model)

    # evaluate!
    start_time = time.time()
    styled_image = stylzr.eval(feed_dict={x: np.array([content_image])})
    print("eval: "+str(time.time() - start_time)+"s")
    styled_image = styled_image.reshape(styled_image.shape[1:])
    utils.write_image(styled_image, OUTPUT_IMAGE_PATH)

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import sys,argparse
import glob
import os
import cv2
#can be installed by running "!pip install opencv-python"
#in current .ipynb
import numpy as np
import time
import pandas as pd
import sys


image_size=128
num_channels=3
num_classes = 3
test_path = sys.argv[1]
#test_path = "../data/training_set/train/"
print test_path
#====== loading data ======
files = [test_path + f for f in os.listdir(test_path) if f.endswith('.jpg')]
images = []
for fl in files:
    image = cv2.imread(fl)
    image = cv2.resize(image, (image_size, image_size), 0, 0, cv2.INTER_LINEAR)
    image = image.astype(np.float32)
    image = np.multiply(image, 1.0 / 255.0)

    images.append(image)

images = np.array(images)
print "all photos read DONE"

#====== Testing on given dataset ======
saver = tf.train.import_meta_graph('../lib/where_are_my_puppies.meta')
sess = tf.Session()
saver.restore(sess, "../lib/./where_are_my_puppies")
graph = tf.get_default_graph()

y_pred = graph.get_tensor_by_name("y_pred:0")
y_pred_cls = tf.argmax(y_pred, axis=1)
x = graph.get_tensor_by_name("x:0")
y_true = graph.get_tensor_by_name("y_true:0")
layer_fc2 = graph.get_tensor_by_name("layer_fc1:0")

results = []
for image in images:
    x_batch = image.reshape(1, image_size, image_size, num_channels)
    y_test_images = np.zeros((1, num_classes))
    feed_dict_testing = {x: x_batch, y_true: y_test_images}
    result = sess.run(layer_fc2, feed_dict=feed_dict_testing)
    results.append(result)


#====== output =====
df = pd.DataFrame(np.resize(np.asarray(results), (len(files),128)))
df.to_csv("../output/feature_CNN_test.csv")

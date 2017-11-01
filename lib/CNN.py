import glob
import os
import cv2
import sys
#can be installed by running "!pip install opencv-python"
#in current .ipynb
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import time
import pandas as pd
from sklearn.utils import shuffle
from datetime import timedelta
import tempfile
import math
import random
#Adding Seed so that random initialization is consistent
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)


num_classes = 3
image_size = 128
validation_size = 0.2
num_channels = 3
batch_size = 64
NUM_SAME_PIC = 3
learning_rate = 0.001
train_path = sys.argv[1]
train_class_path = sys.argv[2]


# ========== reading pictures and classes ==========
class DataSet(object):

    def __init__(self, images, labels, cls):
        self._num_examples = images.shape[0]

        self._images = images
        self._labels = labels
        self._cls = cls
        self._epochs_done = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def cls(self):
        return self._cls

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_done(self):
        return self._epochs_done

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size

        if self._index_in_epoch > self._num_examples:
            # After each epoch we update this
            self._epochs_done += 1
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch

        return self._images[start:end], self._labels[start:end], self._cls[start:end]




# images including all the pictures
def read_train_sets(train_path, train_class_path, image_size, validation_size):
    class DataSets(object):
        pass
    data_sets = DataSets()

    start_time = time.time()

    #==== reading pictures =====
    files = [train_path + f for f in os.listdir(train_path) if f.endswith('.jpg')]
    images = []
    for fl in files:
        image = cv2.imread(fl)
        image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
        image = image.astype(np.float32)
        image = np.multiply(image, 1.0 / 255.0)
        images.append(image)

        #flip
        image_flip = cv2.flip(image,1)
        images.append(image_flip)

        #rotate by -15 ~ +15 degree
        Left = np.random.uniform(-15, 15)
        rows, cols, color = image.shape
        M = cv2.getRotationMatrix2D((cols/2, rows/2), Left, 1)
        image_rotate = cv2.warpAffine(image, M, (cols, rows))
        images.append(image_rotate)

    print("--- reading image part-I DONE %s seconds ---" % (time.time() - start_time))
    images = np.array(images)

    print("--- reading image all DONE %s seconds ---" % (time.time() - start_time))


    #==== reading classes =====
    cls = []
    clsFile = pd.read_csv(train_class_path, index_col=0)
    cls_tmp = clsFile.iloc[:,0].values
    for cur in cls_tmp:
        cls += [cur]*NUM_SAME_PIC
    cls = np.array(cls)

    print("--- reading classes DONE %s seconds ---" % (time.time() - start_time))

    print "Is Label Number = Image Number?", cls.shape[0] == images.shape[0]

    #==== adding labels =====
    labels = []
    for i in cls:
        label = np.zeros(num_classes)
        label[i] = 1.0
        labels.append(label)
    labels = np.array(labels)


    #==== sampling validation out =====
    if isinstance(validation_size, float):
        validation_size = int(validation_size * (images.shape[0]/NUM_SAME_PIC))
    shuffle_idx_tmp = np.random.choice(images.shape[0]/NUM_SAME_PIC, validation_size, replace=False) #3000
    tmp = np.asarray(range(images.shape[0])) #9000
    shuffle_idx = np.reshape([tmp[i * NUM_SAME_PIC : (i+1) * NUM_SAME_PIC] for i in shuffle_idx_tmp], (3*len(shuffle_idx_tmp)))
    # for [0, 7] generate [0, 1, 2, 21, 22, 23]
    not_shuffle_indx = [x for x in tmp if x not in shuffle_idx]

    validation_images = images[shuffle_idx]
    validation_labels = labels[shuffle_idx]
    validation_cls = cls[shuffle_idx]

    train_images = images[not_shuffle_indx]
    train_labels = labels[not_shuffle_indx]
    train_cls = cls[not_shuffle_indx]

    train_images, train_labels, train_cls = shuffle(train_images, train_labels, train_cls)

    print("--- all DONE %s seconds ---" % (time.time() - start_time))


    data_sets.train = DataSet(train_images, train_labels, train_cls)
    data_sets.valid = DataSet(validation_images, validation_labels, validation_cls)

    return data_sets








# ========== load data ==========
data = read_train_sets(train_path, train_class_path, image_size, validation_size=validation_size)
print("Complete reading input data. Will Now print a snippet of it")
print("Number of files in Training-set:\t\t{}".format(len(data.train.cls)))
print("Number of files in Validation-set:\t{}".format(len(data.valid.cls)))




# ========== Placeholders and Parameters ==========
session = tf.Session()
x = tf.placeholder(tf.float32, shape=[None, image_size,image_size,num_channels], name='x')
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true') #labels
y_true_cls = tf.argmax(y_true, axis=1)



##Network graph params
filter_size_conv1 = 3
num_filters_conv1 = 32

filter_size_conv2 = 3
num_filters_conv2 = 32

filter_size_conv3 = 3
num_filters_conv3 = 64

fc_layer_size = 128




# ========== CNN Layers Definition ===========
def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def create_biases(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))

def create_convolutional_layer(input,
               num_input_channels,
               conv_filter_size,
               num_filters,
               layer_name):

    ## We shall define the weights that will be trained using create_weights function.
    weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])
    ## We create biases using the create_biases function. These are also trained.
    biases = create_biases(num_filters)

    ## Creating the convolutional layer
    layer = tf.nn.conv2d(input=input,
                     filter=weights,
                     strides=[1, 1, 1, 1],
                     padding='SAME', name = layer_name+'_conv2d')

    layer += biases

    ## We shall be using max-pooling.
    layer = tf.nn.max_pool(value=layer,
                            ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1],
                            padding='SAME',name = layer_name+'_max_pool')
    ## Output of pooling is fed to Relu which is the activation function for us.
    layer = tf.nn.relu(layer, name = layer_name+'_relu')

    return layer



def create_flatten_layer(layer):
    #We know that the shape of the layer will be [batch_size img_size img_size num_channels]
    # But let's get it from the previous layer.
    layer_shape = layer.get_shape()

    ## Number of features will be img_height * img_width* num_channels. But we shall calculate it in place of hard-coding it.
    num_features = layer_shape[1:4].num_elements()

    ## Now, we Flatten the layer so we shall have to reshape to num_features
    layer = tf.reshape(layer, [-1, num_features])

    return layer


def create_fc_layer(input, num_inputs, num_outputs, layer_name, use_relu=True,):

    #Let's define trainable weights and biases.
    weights = create_weights(shape=[num_inputs, num_outputs])
    biases = create_biases(num_outputs)

    # Fully connected layer takes input x and produces wx+b.Since, these are matrices, we use matmul function in Tensorflow
    layer = tf.matmul(input, weights) + biases
    if use_relu:
        layer = tf.nn.relu(layer, name = layer_name)

    return layer



layer_conv1 = create_convolutional_layer(input=x,
               num_input_channels=num_channels,
               conv_filter_size=filter_size_conv1,
               num_filters=num_filters_conv1,
               layer_name = 'layer_conv1')
layer_conv2 = create_convolutional_layer(input=layer_conv1,
               num_input_channels=num_filters_conv1,
               conv_filter_size=filter_size_conv2,
               num_filters=num_filters_conv2,
               layer_name = 'layer_conv2')

layer_conv3 = create_convolutional_layer(input=layer_conv2,
               num_input_channels=num_filters_conv2,
               conv_filter_size=filter_size_conv3,
               num_filters=num_filters_conv3,
               layer_name = 'layer_conv3')

layer_flat = create_flatten_layer(layer_conv3)

layer_fc1 = create_fc_layer(input=layer_flat,
                     num_inputs=layer_flat.get_shape()[1:4].num_elements(),
                     num_outputs=fc_layer_size,
                     layer_name = 'layer_fc1',
                     use_relu = True)

layer_fc2 = create_fc_layer(input=layer_fc1,
                     num_inputs=fc_layer_size,
                     num_outputs=num_classes,
                     layer_name = 'layer_fc2',
                     use_relu = False)

y_pred = tf.nn.softmax(layer_fc2,name='y_pred')

y_pred_cls = tf.argmax(y_pred, axis=1)


cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
                                                    labels=y_true)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))









# ========== Training Function and Saving to Tensorboard ==========
def train(num_iteration):
    record_best_acc = 0
    record_best_val_acc = 0

    training_summary = tf.summary.scalar("training_accuracy", accuracy)
    validation_summary = tf.summary.scalar("validation_accuracy", accuracy)
    validation_cost = tf.summary.scalar("validation_cost", cost)

    graph_location = tempfile.mkdtemp()
    print('Saving graph to: %s' % graph_location)
    train_writer = tf.summary.FileWriter(graph_location)
    train_writer.add_graph(tf.get_default_graph())

    writer = tf.summary.FileWriter(graph_location, graph=tf.get_default_graph())

    for i in range(num_iteration):

        x_batch, y_true_batch, cls_batch = data.train.next_batch(batch_size)
        x_valid_batch, y_valid_batch, valid_cls_batch = data.valid.next_batch(batch_size)


        feed_dict_tr = {x: x_batch,
                           y_true: y_true_batch}
        feed_dict_val = {x: x_valid_batch,
                              y_true: y_valid_batch}

        session.run(optimizer, feed_dict=feed_dict_tr)


        if i % 200 == 0:
            epoch = int(i / 100)

            #tensorboard
            acc,train_summ = session.run([accuracy,training_summary], feed_dict=feed_dict_tr)
            writer.add_summary(train_summ, i)
            val_acc,val_loss, valid_summ,val_cost = session.run([accuracy, cost, validation_summary,validation_cost], feed_dict=feed_dict_val)
            writer.add_summary(valid_summ, i)
            writer.add_summary(val_cost, i)

            # print output
            #print(val_loss)
            msg = "Training Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%},  Validation Loss: {3:.3f}"
            print(msg.format(epoch + 1, acc, val_acc, val_loss))

            if (val_acc > record_best_val_acc) and (acc > record_best_acc) :
                record_best_acc = acc
                record_best_val_acc = val_acc

                saver.save(session, 'where_are_my_puppies_test00001')



# ========== Training Model ==========
session.run(tf.global_variables_initializer())
saver = tf.train.Saver()
train(num_iteration=5000)

sess.close()

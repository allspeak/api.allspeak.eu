#===========================================================================================================================
# aims      :   define both 3 hidden layers model and 4 hidden layers model
#
# input     :   x : placeholder variable which has as column number the number of features of the training matrix considered
#
#               training_data_len: Number of features of the training matrix considered
#               nclasses: node number of the output layer
#               train_vars_name: set as initnet_vars_name in main_train_net.py or as ftnet_vars_name in 3_main_fine_tuning.py
#
# return    :   x : first layer
#               out_layer : output layer
#===========================================================================================================================

import tensorflow as tf
from . import freeze

n_nodes_hl1=500
n_nodes_hl2=500
n_nodes_hl3=500
n_nodes_hl4=500


#======================================================================================================
# NETWORK: 3 hidden layers and 500 neurons per layer
def create_nn_model3(x, ninputdata_len, nclasses, train_vars_name):

    # with tf.name_scope("inputs"):
    #     x = tf.placeholder(tf.float32, [None, ninputdata_len], name='I')

    with tf.name_scope(train_vars_name):
        W1 = weights([ninputdata_len, n_nodes_hl1], 'W1')
        b1 = biases([n_nodes_hl1], 'b1')
        W2 = weights([n_nodes_hl1, n_nodes_hl2], 'W2')
        b2 = biases([n_nodes_hl2], 'b2')
        W3 = weights([n_nodes_hl2, n_nodes_hl3], 'W3')
        b3 = biases([n_nodes_hl3], 'b3')
        WOUT = weights_out([n_nodes_hl3, nclasses], 'WOUT')
        bOUT = biases_out([nclasses], 'bOUT')

    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, W1), b1)
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, W2), b2)
    layer_2 = tf.nn.relu(layer_2)
    # Hidden layer with RELU activation
    layer_3 = tf.add(tf.matmul(layer_2, W3), b3)
    layer_3 = tf.nn.relu(layer_3)
    # Output layer with linear activation
    out_layer = tf.add(tf.matmul(layer_3, WOUT), bOUT, name='O')
    soft_layer = tf.nn.softmax(out_layer, name='SMO')

    return {'input': x, 'output': out_layer}


def create_nn_model4(x, ninputdata_len, noutputclasses, train_vars_name, graph):

    W2 = graph.get_tensor_by_name('prefix/model/W2:0')
    W3 = graph.get_tensor_by_name('prefix/model/W3:0')
    b2 = graph.get_tensor_by_name('prefix/model/b2:0')
    b3 = graph.get_tensor_by_name('prefix/model/b3:0')

    with tf.variable_scope(train_vars_name):
        WOUT1p = weights_out([n_nodes_hl3, noutputclasses], 'WOUT1p')
        bOUT1p = biases_out([noutputclasses], 'bOUT1p')
        W4 = weights([n_nodes_hl3, n_nodes_hl4], 'W4')
        b4 = biases([n_nodes_hl4], 'b4')
        W11p = weights([ninputdata_len, n_nodes_hl1], 'W11p')
        b11p = biases([n_nodes_hl1], 'b11p')

    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, W11p), b11p)
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, W2), b2)
    layer_2 = tf.nn.relu(layer_2)
    # Hidden layer with RELU activation
    layer_3 = tf.add(tf.matmul(layer_2, W3), b3)
    layer_3 = tf.nn.relu(layer_3)
    # Hidden layer with RELU activation
    layer_4 = tf.add(tf.matmul(layer_3, W4), b4)
    layer_4 = tf.nn.relu(layer_4)
    # Output layer with linear activation
    out_layer = tf.add(tf.matmul(layer_4, WOUT1p), bOUT1p, name='O')
    soft_layer = tf.nn.softmax(out_layer, name='SMO')
    return {'input': x, 'output': out_layer}


def weights(shape,name):
  initial = tf.truncated_normal(shape,stddev=0.1)
  return tf.Variable(initial, name=name)


def weights_out(shape,name):
  initial = tf.truncated_normal(shape,stddev=0.1)
  return tf.Variable(initial, name=name)


def biases(shape,name):
  initial = tf.constant(0.1,shape=shape)
  return tf.Variable(initial, name=name)


def biases_out(shape,name):
  initial = tf.constant(0.1,shape=shape)
  return tf.Variable(initial, name=name)


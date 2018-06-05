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
from . import utilities

n_nodes_hl1=500
n_nodes_hl2=500
n_nodes_hl3=500
n_nodes_hl4=500


#======================================================================================================
# I CREATE A 3HL FF MODEL
#======================================================================================================
# NETWORK: 3 hidden layers and 500 neurons per layer
def new_ff_model3(x, ninputdata_len, nclasses, train_vars_name):

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

#======================================================================================================
# I CREATE A 4HL FF MODEL
#======================================================================================================
# NETWORK: 4 hidden layers and 500 neurons per layer
def new_ff_model4(x, ninputdata_len, nclasses, train_vars_name):

    # with tf.name_scope("inputs"):
    #     x = tf.placeholder(tf.float32, [None, ninputdata_len], name='I')

    with tf.name_scope(train_vars_name):
        W1 = weightsUniform([ninputdata_len, n_nodes_hl1], 'W1')
        b1 = biases([n_nodes_hl1], 'b1')
        W2 = weightsUniform([n_nodes_hl1, n_nodes_hl2], 'W2')
        b2 = biases([n_nodes_hl2], 'b2')
        W3 = weightsUniform([n_nodes_hl2, n_nodes_hl3], 'W3')
        b3 = biases([n_nodes_hl3], 'b3')
        W4 = weightsUniform([n_nodes_hl3, n_nodes_hl4], 'W4')
        b4 = biases([n_nodes_hl4], 'b4')
        WOUT = weightsUniform_out([n_nodes_hl4, nclasses], 'WOUT')
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
    # Hidden layer with RELU activation
    layer_4 = tf.add(tf.matmul(layer_3, W4), b4)
    layer_4 = tf.nn.relu(layer_4)
    # Output layer with linear activation
    out_layer = tf.add(tf.matmul(layer_4, WOUT), bOUT, name='O')
    soft_layer = tf.nn.softmax(out_layer, name='SMO')

    return {'input': x, 'output': out_layer}

# ======================================================================================================
# I ADAPT AN EXISTING 3HL FF NET => 4HL NET
# ======================================================================================================
# I inherit from a graph the 2nd & 3rd hidden layers' weights
# I create a 4th hidden layer
# I will train the latter + the 1st and the OutLayer
def adapt_ff_model3(x, ninputdata_len, noutputclasses, train_vars_name, graph, prefix=None):

    if prefix is not None:
        prefix = prefix + "/"
    else:
        prefix = ""

    nodes = [n.name for n in graph.as_graph_def().node]

    W2 = utilities.getNodeBySubstring(graph, prefix + 'model/W2', nodes)
    W3 = utilities.getNodeBySubstring(graph, prefix + 'model/W3', nodes)
    b2 = utilities.getNodeBySubstring(graph, prefix + 'model/b2', nodes)
    b3 = utilities.getNodeBySubstring(graph, prefix + 'model/b3', nodes)

    with tf.variable_scope(train_vars_name):
        W11p = weights([ninputdata_len, n_nodes_hl1], 'W11p')
        b11p = biases([n_nodes_hl1], 'b11p')
        W4 = weights([n_nodes_hl3, n_nodes_hl4], 'W4')
        b4 = biases([n_nodes_hl4], 'b4')
        WOUT1p = weights_out([n_nodes_hl3, noutputclasses], 'WOUT1p')
        bOUT1p = biases_out([noutputclasses], 'bOUT1p')

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

# ======================================================================================================
# I RE-ADAPT AN EXISTING 4HL FF NET => 4HL NET
# ======================================================================================================
# A)
# I inherit from a graph the 2nd & 3rd & 4th hidden layers' weights
# I will train first and last layer
def readapt_ff_adaptedmodel(x, ninputdata_len, noutputclasses, train_vars_name, graph, prefix=None):

    if prefix is not None:
        prefix = prefix + "/"
    else:
        prefix = ""

    nodes = [n.name for n in graph.as_graph_def().node]

    W2 = utilities.getNodeBySubstring(graph, prefix + 'model/W2', nodes)
    W3 = utilities.getNodeBySubstring(graph, prefix + 'model/W3', nodes)
    W4 = utilities.getNodeBySubstring(graph, prefix + train_vars_name + '/W4', nodes)

    b2 = utilities.getNodeBySubstring(graph, prefix + 'model/b2', nodes)
    b3 = utilities.getNodeBySubstring(graph, prefix + 'model/b3', nodes)
    b4 = utilities.getNodeBySubstring(graph, prefix + train_vars_name + '/b4', nodes)

    with tf.variable_scope(train_vars_name):
        W11p = weights([ninputdata_len, n_nodes_hl1], 'W11p')
        b11p = biases([n_nodes_hl1], 'b11p')
        WOUT1p = weights_out([n_nodes_hl3, noutputclasses], 'WOUT1p')
        bOUT1p = biases_out([noutputclasses], 'bOUT1p')

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

# B)
# I inherit all the existing weights (nodes names obtained according to : adapt_ff_model3
# I train everything
def readapt_ff_adaptedmodel_2(x, ninputdata_len, noutputclasses, train_vars_name, graph, prefix=None):

    if prefix is not None:        prefix = prefix + "/"

    b11p = graph.get_tensor_by_name('model/b11p:0')
    b2 = graph.get_tensor_by_name('model/b2:0')
    b3 = graph.get_tensor_by_name('model/b3:0')
    b4 = graph.get_tensor_by_name('model/b4:0')
    bOUT1p = graph.get_tensor_by_name('model/bOUT1p:0')

    # with tf.variable_scope(train_vars_name):
    #     W11p = weights([ninputdata_len, n_nodes_hl1], 'W11p')
    #     b11p = biases([n_nodes_hl1], 'b11p')
    #     WOUT1p = weights_out([n_nodes_hl3, noutputclasses], 'WOUT1p')
    #     bOUT1p = biases_out([noutputclasses], 'bOUT1p')

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


# ======================================================================================================
# ======================================================================================================
def weights(shape, name):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial, name=name)


def weights_out(shape, name):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial, name=name)


def weightsUniform(shape, name):
  initial = tf.random_uniform(shape, minval=-0.1, maxval=0.1)
  return tf.Variable(initial, name=name)


def weightsUniform_out(shape, name):
  initial = tf.random_uniform(shape, minval=-0.1, maxval=0.1)
  return tf.Variable(initial, name=name)


def biases(shape,name):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial, name=name)


def biases_out(shape,name):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial, name=name)


# This module is related to the training phase and it contains all the functions listed below:
#
# createSubjectTrainingMatrix(subj, in_orig_subj_path, output_net_path, arr_commands, arr_rip)
# createSubjectTestMatrix(subj, in_orig_subj_path, output_net_path, arr_commands, arr_rip, sentences_filename, sentence_counter)
# createFullMatrix(input_matrix_folder, data_name, label_name, output_matrix_path="")
# train(sess, saver, training_data, training_label, train_vars_name, input_layer, output_layer, y, output_dir, net_name, hm_epochs, nFrames, batch_size)
# fineTuning(subj, commands_list, input_model_name, train_repetitions_list, test_repetitions_list, sentence_counter_filename, output_net_root_path, test_data_root_path, subjects_root_path, subj_outnet_name, clean_folder=True, hm_epochs=20, batch_size=100, ftnet_vars_name="fine_tuning_weights")


import tensorflow as tf
import numpy as np
import re
import glob
import os
import time
import shutil
import zipfile
import json
from numpy import genfromtxt
from . import freeze
from . import models
from . import utilities
from . import context
from pathlib import Path


# ===========================================================================================================================
# aims      :   This script creates the training matrix FROM a list of subjects
#
# input     :   subjects_list               ["a","b","c","d"]
#               output_model_name           controlsNOemanuele_spectfiltthresh
#               commands_list :             ids of commands to be trained
#               train_repetitions_list, test_repetitions_list:  indicates repetitions' ids of files to be used for either test or training  [3,4,5,6,7,8,9] [0,1,2]
#               train_output_net_path, test_output_net_path
#               subjects_root_path
#               sentence_counter_filename
#               initnet_vars_name="model", batch_size=100, hm_epochs=20, clean_folder=True
# return    :   output_matrices_path: path to the output folder
# ===========================================================================================================================
def trainSubjects(subjects_list, output_model_name, commands_list, training_repetitions_list, test_repetitions_list,
                  train_output_net_path, test_output_net_path, subjects_root_path, sentence_counter_filename, voicebank_vocabulary_path, doOnlyTrain=True,
                  initnet_vars_name="model", batch_size=100, hm_epochs=20, clean_folder=True):

    ncommands = len(commands_list)

    if os.path.isdir(train_output_net_path) is False:
        os.mkdir(train_output_net_path)

    if os.path.isdir(test_output_net_path) is False and doOnlyTrain is False:
        os.mkdir(test_output_net_path)
    sentence_counter = 0

    for subj in subjects_list:
        in_orig_subj_path = subjects_root_path + '/' + subj

        training_matrices_output = createSubjectTrainingMatrix(subj, in_orig_subj_path, train_output_net_path,
                                                                     commands_list, training_repetitions_list)
        if doOnlyTrain is False:
            test_matrices_output = createSubjectTestMatrix(subj, in_orig_subj_path, test_output_net_path,
                                                             commands_list, test_repetitions_list,
                                                             sentence_counter_filename, sentence_counter)
            sentence_counter = test_matrices_output['sentence_counter']


    train_matrices = createFullMatrix(training_matrices_output['matrices_path'], 'train_data', 'train_labels')
    if doOnlyTrain is False:
        test_matrices = createFullMatrix(test_matrices_output['matrices_path'], 'test_data', 'test_labels', test_output_net_path)

    # train_matrices = {'data_matrix_path': '/data/AllSpeak/CODE/tf/output/pretraining_nets/allcontrols_spectfiltthresh/matrices/full_train_data.npy', 'label_matrix_path': '/data/AllSpeak/CODE/tf/output/pretraining_nets/allcontrols_spectfiltthresh/matrices/full_train_labels.npy'}
    train_data_matrix = np.load(train_matrices['data_matrix_path'])
    train_label_matrix = np.load(train_matrices['label_matrix_path'])

    nFrames = train_data_matrix.shape[0]  # righe: frames
    training_data_len = len(train_data_matrix[0])  # colonne: input layer length

    with tf.name_scope("inputs"):
        x = tf.placeholder(tf.float32, [None, training_data_len], name='I')
        y = tf.placeholder(tf.float32, [None, ncommands], name='y-input')

    net_inout = models.create_nn_model3(x, training_data_len, ncommands, initnet_vars_name)
    input_layer = net_inout["input"]
    out_layer = net_inout["output"]

    # 'Saver' op to save and restore first 3 layers
    saver = tf.train.Saver()

    # Running first session
    print("Starting 1st session...")
    with tf.Session() as sess:
        with tf.device('/cpu:0'):
            start_time = time.time()
            sess = train(sess, saver, train_data_matrix, train_label_matrix, initnet_vars_name, input_layer,
                               out_layer, y, train_output_net_path, output_model_name, nFrames, hm_epochs, batch_size)
            elapsed_time = time.time() - start_time
            print("elapsed time : " + str(elapsed_time))

        with tf.device('/cpu:0'):
            result = freeze.freeze(output_model_name, train_output_net_path, "SMO")

            output_optimized_graph_name = result['grp_opt_name']
            if clean_folder is True and os.path.isdir(train_output_net_path + "/matrices") is True:
                shutil.rmtree(train_output_net_path + "/matrices")  # /matrices is defined in createSubjectMatrix

    utilities.createVocabularySentence(commands_list, voicebank_vocabulary_path, train_output_net_path + '/vocabulary.txt')


# ===========================================================================================================================
# aims      :   alternative version of the above methods. reuse subjects matrices. assemble a new full_matrix_data and train it
#
# input     :   subjects_list
#               output_model_name
#               train_data_input_matrices_path          : folder containing subjects' data matrices
#               train_labels_input_matrices_path        : folder containing subjects' labels matrices
#               commands_list : ids of commands to be trained
#               train_repetitions_list, test_repetitions_list:  indicates repetitions' ids of files to be used for either test or training  [3,4,5,6,7,8,9] [0,1,2]
#               train_output_net_path, test_output_net_path : output folder of net and test data
#               sentence_counter_filename
#               initnet_vars_name="model", batch_size=100, hm_epochs=20, clean_folder=True
# return    :   output_matrices_path: path to the output folder
# ===========================================================================================================================
def trainExistingSubjects(subjects_list, output_model_name, train_data_input_matrices_path, train_labels_input_matrices_path,
                          commands_list, training_repetitions_list, test_repetitions_list,
                          train_output_net_path, test_output_net_path, sentence_counter_filename, voicebank_vocabulary_path, doOnlyTrain=True,
                          initnet_vars_name="model", batch_size=100, hm_epochs=20, clean_folder=False):

    ncommands = len(commands_list)

    if os.path.isdir(train_output_net_path) is False:
        os.mkdir(train_output_net_path)

    if os.path.isdir(test_output_net_path) is False:
        os.mkdir(test_output_net_path)

    train_matrices = createFullMatrix2(train_data_input_matrices_path, subjects_list, 'train_data', 'train_labels', train_output_net_path)
    test_matrices = createFullMatrix2(train_labels_input_matrices_path, subjects_list, 'test_data', 'test_labels', test_output_net_path)

    # sentences_file.close()
    train_data_matrix = np.load(train_matrices['data_matrix_path'])
    train_label_matrix = np.load(train_matrices['label_matrix_path'])

    nFrames = train_data_matrix.shape[0]  # righe: frames
    training_data_len = len(train_data_matrix[0])  # colonne: input layer length

    with tf.name_scope("inputs"):
        x = tf.placeholder(tf.float32, [None, training_data_len], name='I')
        y = tf.placeholder(tf.float32, [None, ncommands], name='y-input')

    net_inout = models.create_nn_model3(x, training_data_len, ncommands, initnet_vars_name)
    input_layer = net_inout["input"]
    out_layer = net_inout["output"]

    # 'Saver' op to save and restore first 3 layers
    saver = tf.train.Saver()

    # Running first session
    print("Starting 1st session...")
    with tf.Session() as sess:
        with tf.device('/cpu:0'):
            start_time = time.time()
            sess = train(sess, saver, train_data_matrix, train_label_matrix, initnet_vars_name, input_layer,
                         out_layer, y, train_output_net_path, output_model_name, nFrames, hm_epochs,
                         batch_size)
            elapsed_time = time.time() - start_time
            print("elapsed time : " + str(elapsed_time))

        with tf.device('/cpu:0'):
            result = freeze.freeze(output_model_name, train_output_net_path, "SMO")

            output_optimized_graph_name = result['grp_opt_name']
            if clean_folder is True and os.path.isdir(train_output_net_path + "/matrices") is True:
                shutil.rmtree(train_output_net_path + "/matrices")  # /matrices is defined in createSubjectMatrix

    utilities.createVocabularySentence(commands_list, voicebank_vocabulary_path, train_output_net_path + '/vocabulary.txt')


# ===========================================================================================================================
# aims      :   This script creates the training matrix for a single subject
#
# input     :   subj: subject folder name
#               in_orig_subj_path: path to the subject's cepstra with context
#               output_net_path: path to the output folder
#               arr_commands: IDs of the selected commands
#               arr_rip: range from 0 to Nripetitions
#
# return    :   output_matrices_path: path to the output folder
# ===========================================================================================================================
def createSubjectTrainingMatrix(subj, in_orig_subj_path, output_net_path, arr_commands, arr_rip):
    mat_compl = []
    mat_lab = []
    output_matrices_path = os.path.join(in_orig_subj_path, 'matrices')

    if os.path.isdir(output_matrices_path) is False:
        os.mkdir(output_matrices_path)

    for ctxfile in glob.glob(in_orig_subj_path + '/ctx*'):
        #print(ctxfile)
        spl = re.split('[_ .]', ctxfile)  # e.g. ctx_SUBJ_CMD_REP => spl2[2] num comando, spl[3] num ripetiz
        id_cmd = int(spl[2])
        id_rep = int(spl[3])

        if id_cmd in arr_commands and id_rep in arr_rip:
            f = open(ctxfile, 'r')
            lines = f.readlines()
            count_lines = len(lines)
            f.close()
            # for every line of contexted file, write N-arr_commands columns
            lb = [[1 if i == id_cmd else 0 for i in arr_commands] for j in range(count_lines)]
            ctx = genfromtxt(ctxfile)  # load dei cepstra

            if len(mat_compl) == 0 and len(mat_lab) == 0:
                mat_compl = ctx
                mat_lab = lb
            else:
                mat_compl = np.vstack((mat_compl, ctx))
                mat_lab = np.vstack((mat_lab, lb))

    # print data
    np.savetxt(output_matrices_path + '/' + subj + '_train_data.dat', mat_compl, fmt='%.4f')
    np.savetxt(output_matrices_path + '/' + subj + '_train_labels.dat', mat_lab, fmt='%.0f')

    print("createSubjectTrainingMatrix ended: "  + str(mat_compl.size))

    return {'matrices_path': output_matrices_path, 'mat_compl': mat_compl, 'mat_lab': mat_lab}

# ===========================================================================================================================
# aims      :   This script creates the testing matrix for a single subject
#
# input     :   subj: subject folder name
#               in_orig_subj_path: path to the subject's cepstra with context
#               output_net_path: path to the output folder
#               arr_commands: range from 1 to Ncommands
#               arr_rip: range from 0 to Nripetitions
#               sentences_filename: name of the output file
#               sentence_counter: it takes account of how many rows are occupied by each command and the command_id
#
# return    :   output_matrices_path: path to the output folder
#               sentence_counter: text file which takes account of how many rows are occupied by each command and the command_id
# ===========================================================================================================================

def createSubjectTestMatrix(subj, in_orig_subj_path, output_net_path, arr_commands, arr_rip, sentences_filename, sentence_counter):
    mat_compl = []
    mat_lab = []
    output_matrices_path = output_net_path + '/matrices'

    if os.path.isdir(output_matrices_path) is False:
        os.mkdir(output_matrices_path)

    if os.path.isfile(sentences_filename) is True:
        os.remove(sentences_filename)

    for ctxfile in glob.glob(in_orig_subj_path + '/ctx*'):

        spl = re.split('[_ .]', ctxfile)  # e.g. ctx_SUBJ_CMD_REP => spl2[2] num comando, spl[3] num ripetiz
        id_cmd = int(spl[2])
        id_rep = int(spl[3])

        if id_cmd in arr_commands and id_rep in arr_rip:

            f = open(ctxfile, 'r')
            lines = f.readlines()
            count_lines = len(lines)
            f.close()

            sentence_counter = sentence_counter + 1
            sc = [[sentence_counter, id_cmd] for j in range(count_lines)]
            with open(output_net_path + "/" + sentences_filename, 'ab') as f_handle:
                np.savetxt(f_handle, sc, fmt='%.0f')

            lb = [[1 if i == id_cmd else 0 for i in arr_commands] for j in range(count_lines)]
            ctx = genfromtxt(ctxfile)  # load dei cepstra

            if len(mat_compl) == 0 and len(mat_lab) == 0:
                mat_compl = ctx
                mat_lab = lb
            else:
                mat_compl = np.vstack((mat_compl, ctx))
                mat_lab = np.vstack((mat_lab, lb))

    # print data
    np.savetxt(output_matrices_path + '/' + subj + '_test_data.dat', mat_compl, fmt='%.4f')
    np.savetxt(output_matrices_path + '/' + subj + '_test_labels.dat', mat_lab, fmt='%.0f')
    return {'matrices_path': output_matrices_path, 'sentence_counter': sentence_counter, 'mat_compl': mat_compl, 'mat_lab': mat_lab}

# ===========================================================================================================================
# aims      :   This script creates the testing matrix with all the pre-established subjects
#
# input     :   input_matrix_folder: path to the subject's folder containing testing and training matrices with cepstra or labels
#               data_name: name of the testing or training matrices with cepstra
#               label_name: name of the testing or training matrices with labels
#               output_matrix_path: path to the output folder. If is not specified, data will be stored in the current working folder
#
# return    :   data_matrix_path: path to the output folder
#               label_matrix_path: path to the output folder
# ===========================================================================================================================

def createFullMatrix(input_matrix_folder, data_name, label_name, output_matrix_path=""):

    train = []
    labels = []

    if len(output_matrix_path):
        data_matrix_path = output_matrix_path + '/full_' + data_name
        label_matrix_path = output_matrix_path + '/full_' + label_name
    else:
        data_matrix_path = input_matrix_folder + '/full_' + data_name
        label_matrix_path = input_matrix_folder + '/full_' + label_name

    for file in glob.glob(input_matrix_folder + '/*' + data_name + '.dat'):
        file_name = os.path.basename(file)
        file_train = genfromtxt(file)
        spl2 = re.split('[_ .]', file_name)  # spl2[0] paz, spl[1] parola 'train'
        file_labels = genfromtxt(input_matrix_folder + '/' + spl2[0] + '_' + label_name + '.dat')
        if len(train) == 0 and len(labels) == 0:
            train = file_train
            labels = file_labels
        else:
            train = np.vstack((train, file_train))
            labels = np.vstack((labels, file_labels))

    # np.savetxt(data_matrix_path, train, fmt='%.4f')
    # np.savetxt(label_matrix_path, labels, fmt='%.0f')
    np.save(data_matrix_path, train)
    np.save(label_matrix_path, labels)

    return {'data_matrix_path': data_matrix_path + '.npy', 'label_matrix_path': label_matrix_path+ '.npy'}


def createFullMatrix2(input_matrix_folder, subjects_list, data_name, label_name, output_matrix_path=""):

    train = []
    labels = []

    if len(output_matrix_path):
        data_matrix_path = output_matrix_path + '/full_' + data_name
        label_matrix_path = output_matrix_path + '/full_' + label_name
    else:
        data_matrix_path = input_matrix_folder + '/full_' + data_name
        label_matrix_path = input_matrix_folder + '/full_' + label_name

    # for file in glob.glob(input_matrix_folder + '/*' + data_name + '.dat'):
    for subject in subjects_list:

        subjfile = input_matrix_folder + "/" + subject + "_" + data_name + ".dat"
        file_name = os.path.basename(subjfile)
        file_train = genfromtxt(subjfile)
        spl2 = re.split('[_ .]', file_name)  # spl2[0] paz, spl[1] parola 'train'
        file_labels = genfromtxt(input_matrix_folder + '/' + spl2[0] + '_' + label_name + '.dat')
        if len(train) == 0 and len(labels) == 0:
            train = file_train
            labels = file_labels
        else:
            train = np.vstack((train, file_train))
            labels = np.vstack((labels, file_labels))

    np.save(data_matrix_path, train)
    np.save(label_matrix_path, labels)

    return {'data_matrix_path': data_matrix_path + '.npy', 'label_matrix_path': label_matrix_path+ '.npy'}


# ===========================================================================================================================
# aims      :   This script allows the 3 hidden layers neural network training and it saves the session
#
# input     :   sess: session currently running
#               training_data: training matrix with cepstra
#               training_label: training matrix with labels
#               train_vars_name: it saves variables under this name
#               input_layer: layer named as "I" in 2_main_train_net.py
#               output_layer: softmax layer
#               y: placeholder variable which has as column number the number of commands
#               output_dir: path to the output folder
#               net_name: output folder name
#               hm_epochs: number of epochs
#               nFrames: training matrix's number of rows
#               batch_size: size of batch
#
# return    :   sess: it saves the session currently running
# ===========================================================================================================================


def train(sess, saver, training_data, training_label, train_vars_name, input_layer, output_layer, y, output_dir, net_name, nframes, hm_epochs=20, batch_size=100):

    # COST FUNCTION:
    cost_ap = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_layer, labels=y))
    train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, train_vars_name)
    optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost_ap, var_list=train_vars)

    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    # save the graph
    # tf.train.write_graph(sess.graph_def, output_dir, "graph.pb", False)
    tf.train.write_graph(sess.graph_def, output_dir, net_name + '.pbtxt')

    # START TRAINING
    for epoch in range(hm_epochs):

        permS = np.random.permutation(nframes)
        epoch_loss = 0

        for iter in range(0, nframes - batch_size - 1, batch_size):
            batch_x = training_data[permS[iter:iter + batch_size], :]
            batch_y = training_label[permS[iter:iter + batch_size], :]
            _, c = sess.run([optimizer, cost_ap], feed_dict={input_layer: batch_x, y: batch_y})
            epoch_loss += c

        print('Epoch', epoch + 1, 'completed out of', hm_epochs, 'loss:', epoch_loss)

    print("First Optimization Finished!")

    saver.save(sess, output_dir + '/' + net_name + '.ckpt')
    return sess


# ===========================================================================================================================
# aims      :   This script allows the 4 hidden layers neural network training, it saves the session and freezes the model in a pb file
#
# input     :   subj: subject name
#               commands_list: range from 1 to Ncommands
#               input_model_name: folder name where it loads the pretrained neural network
#               train_repetitions_list: range from 0 to Nripetitions
#               test_repetitions_list: range from 0 to Nripetitions. It should be complementary to the train_repetitions_list
#               sentence_counter_filename: name of the output file
#               output_net_root_path: general path to all the pretraining nets
#               test_data_root_path: general path to all the testing folders
#               subjects_root_path: general path to all the subject's folder containing cepstra
#               subj_outnet_name: folder named as subj_outnet_name in 3_main_fine_tuning.py
#               hm_epochs: number of epochs
#               batch_size: size of batch
#
# return    :   void
# ===========================================================================================================================

def fineTuning(subj, commands_list, input_model_name, training_repetitions_list, test_repetitions_list, sentence_counter_filename, output_net_root_path, test_data_root_path, subjects_root_path, subj_outnet_name, clean_folder=True, hm_epochs=20, batch_size=100, ftnet_vars_name="fine_tuning_weights"):

    train_output_net_path = output_net_root_path + '/' + subj_outnet_name
    test_output_net_path = test_data_root_path + '/' + subj_outnet_name

    input_initmodel_path = output_net_root_path + "/" + input_model_name + "/optimized_" + input_model_name + ".pb"
    output_ftmodel_path = train_output_net_path + "/optimized_" + subj_outnet_name + ".pb"

    ncommands = len(commands_list)

    if os.path.isdir(train_output_net_path) is False:
        os.mkdir(train_output_net_path)

    if os.path.isdir(test_output_net_path) is False:
        os.mkdir(test_output_net_path)

    sentence_counter = 0

    # -------------------------------------------------------------------
    # create subjects' matrix
    in_orig_subj_path = subjects_root_path + '/' + subj
    training_matrices_output = createSubjectTrainingMatrix(subj, in_orig_subj_path, train_output_net_path,
                                                                 commands_list, training_repetitions_list)

    test_matrices_output = createSubjectTestMatrix(subj, in_orig_subj_path, test_output_net_path, commands_list,
                                                         test_repetitions_list, sentence_counter_filename,
                                                         sentence_counter)
    sentence_counter = test_matrices_output['sentence_counter']

    train_data_matrix = training_matrices_output['mat_compl']
    train_label_matrix = training_matrices_output['mat_lab']

    # train_data_matrix = genfromtxt(training_matrices_output['matrices_path'] + "/" + subj + "_train_data.dat")
    # train_label_matrix = genfromtxt(training_matrices_output['matrices_path'] + "/" + subj + "_train_labels.dat")

    Nexe = train_data_matrix.shape[0]
    ninputdata_len = len(train_data_matrix[0])

    graph = freeze.load(input_initmodel_path)

    with tf.Session(graph=graph) as sess:
        with tf.name_scope("inputs"):
            x = tf.placeholder(tf.float32, [None, ninputdata_len], name='I')
            y = tf.placeholder(tf.float32, [None, ncommands], name='y-input')

        net_inout = models.create_nn_model4(x, ninputdata_len, ncommands, ftnet_vars_name, graph)
        input_layer = net_inout["input"]
        out_layer = net_inout["output"]

        # 'Saver' op to save and restore first 3 layers
        saver = tf.train.Saver()

        # Running first session
        print("Starting 1st session...")

        with tf.device('/cpu:0'):
            start_time = time.time()
            sess = train(sess, saver, train_data_matrix, train_label_matrix, ftnet_vars_name, input_layer,
                               out_layer, y, train_output_net_path, subj_outnet_name, Nexe, hm_epochs, batch_size)
            elapsed_time = time.time() - start_time
            print("elapsed time : " + str(elapsed_time))

        with tf.device('/cpu:0'):
            result = freeze.freeze(subj_outnet_name, train_output_net_path, "SMO")
            output_optimized_graph_name = result['grp_opt_name']
            if clean_folder is True:
                shutil.rmtree(train_output_net_path + "/matrices")  # /matrices is defined in createSubjectMatrix


# send all data to training, does not create any test data.
# : inputdata_folder. folder containing ctx_xxxxxx.dat files
# : output_net_path. where to save the FT NET
# : output_net_name. name of the output pb file
#def fineTuningFolderOnlyTrain(inputdata_folder, commands_list, input_model_path, output_net_path, output_net_name, clean_folder=True, hm_epochs=20, batch_size=100, ftnet_vars_name="fine_tuning_weights"):
def fineTuningFolderOnlyTrain(inputdata_folder, commands_list, output_net_path, output_net_name, train_data, clean_folder=True):

    batch_size = train_data['batch_size']           # 
    hm_epochs = train_data['hm_epochs']             # 
    ftnet_vars_name = train_data['ftnet_vars_name'] # fine_tuning_weights
    sInputNodeName = train_data['sInputNodeName']   # inputs/I unused
    sOutputNodeName = train_data['sOutputNodeName'] # 
    init_net_path = train_data['init_net_path']     #

    session_dir = os.path.dirname(inputdata_folder)
    output_ftmodel_path = session_dir + "/optimized_" + output_net_name + ".pb"
    ncommands = len(commands_list)

    if os.path.isdir(output_net_path) is False:
        os.mkdir(output_net_path)

    # -------------------------------------------------------------------
    # create subjects' matrix
    training_matrices_output = createSubjectTrainingMatrix("", inputdata_folder, output_net_path, commands_list, range(0, 25))

    train_data_matrix = training_matrices_output['mat_compl']
    train_label_matrix = training_matrices_output['mat_lab']

    # train_data_matrix = genfromtxt(training_matrices_output['matrices_path'] + "/" + "user" + "_train_data.dat")
    # train_label_matrix = genfromtxt(training_matrices_output['matrices_path'] + "/" + "user" + "_train_labels.dat")

    Nexe = train_data_matrix.shape[0]
    ninputdata_len = len(train_data_matrix[0])


    graph = freeze.load(init_net_path)

    with tf.Session(graph=graph) as sess:
        with tf.name_scope("inputs"):
            x = tf.placeholder(tf.float32, [None, ninputdata_len], name='I')
            y = tf.placeholder(tf.float32, [None, ncommands], name='y-input')

        net_inout = models.create_nn_model4(x, ninputdata_len, ncommands, ftnet_vars_name, graph)
        input_layer = net_inout["input"]
        out_layer = net_inout["output"]

        # 'Saver' op to save and restore first 3 layers
        saver = tf.train.Saver()

        # Running first session
        print("Starting 1st session...")

        with tf.device('/cpu:0'):
            start_time = time.time()
            sess = train(sess, saver, train_data_matrix, train_label_matrix, ftnet_vars_name, input_layer,
                               out_layer, y, output_net_path, output_net_name, Nexe, hm_epochs, batch_size)
            elapsed_time = time.time() - start_time
            print("elapsed time : " + str(elapsed_time))

        with tf.device('/cpu:0'):
            result = freeze.freeze(output_net_name, output_net_path, sOutputNodeName)
            output_optimized_graph_name = result['grp_opt_name']

            print(output_optimized_graph_name + "," + output_ftmodel_path)
            os.rename(output_optimized_graph_name, output_ftmodel_path)
            if clean_folder is True:
                shutil.rmtree(inputdata_folder)  # /matrices is defined in createSubjectMatrix


def train_net(training_sessionid, user_id, modeltype, commands_ids, str_proc_scheme, clean_folder=True):

    #print(str(training_sessionid) + " " + str(user_id) + " " + str(modeltype) + " " + str_proc_scheme + " " + str(len(commands_ids)) + " ")
    folder_path = os.path.join('project', 'data', str(training_sessionid))
    lockfile_path = os.path.join(folder_path, '.lock')
    Path(lockfile_path).touch()


    if modeltype == 274:
        trainparams_json = os.path.join('project', 'training_api', 'train_params.json')    
    else:
        trainparams_json = os.path.join('project', 'training_api', 'ft_train_params.json')
    
    print(os.getcwd())
    #a = os.path.join('project', 'training_api', 'train_params.json')
    print(trainparams_json)
    
    #os.path.join('project', 'training_api', 'train_params.json')


    with open(trainparams_json, 'r') as data_file:
        train_data = json.load(data_file)

    print(train_data['nContextFrames'])
    ctx_frames = train_data['nContextFrames']           # 
    sModelFileName = train_data['sModelFileName']   # 

    output_net_name = "%s_%d_%s" % (sModelFileName, user_id, str_proc_scheme)
    output_net_path = os.path.join(folder_path, 'data', 'net')
    data_path = os.path.join(folder_path, 'data')


    # CONTEXTING DATA (create ctx_...  files)
    context.createSubjectContext(data_path, ctx_frames)

    if modeltype == 274:
        fineTuningFolderOnlyTrain(data_path, commands_ids, output_net_path, output_net_name, train_data, clean_folder)
    else:
        fineTuningFolderOnlyTrain(data_path, commands_ids, output_net_path, output_net_name, train_data, clean_folder)

    print('tuning done')
    os.remove(lockfile_path)

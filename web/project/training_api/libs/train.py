# This module is related to the training phase and it contains all the functions listed below:
# train_net 
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
import random
from numpy import genfromtxt
from . import freeze
from . import models
from . import utilities
from . import context
from project import db
from project.models import TrainingSession, User

# =========================================================================================================================
# =========================================================================================================================
# entry point of all trainings (called by views.py : @training_api_blueprint.route('/api/v1/training-sessions')
# =========================================================================================================================
# =========================================================================================================================
# session_path is   : instance/users_data/APIKEY/train_data/training_sessionid
# data files are in : session_path/data
# session_data is   : nModelType, nProcessingScheme, commands, init_sessionid
def train_net(session_data, session_path, training_sessionid, voicebank_vocabulary_path, clean_folder=True):
    
    try:
        # GET SESSION INFO
        modeltype = session_data['nModelType']
        str_proc_scheme = str(session_data['nProcessingScheme'])  # 252/253/254/255/256/257/258
        commands = session_data['commands']
        commands_ids = [cmd['id'] for cmd in commands]

        # LOAD SELECTED MODEL PARAMS
        if modeltype == 274:
            trainparams_json = os.path.join('project', 'training_api', 'pure_user_trainparams.json')
        elif modeltype == 275:
            trainparams_json = os.path.join('project', 'training_api', 'pure_user_adapted_trainparams.json')    
        elif modeltype == 276:
            trainparams_json = os.path.join('project', 'training_api', 'common_adapted_trainparams.json')    
        elif modeltype == 277:
            trainparams_json = os.path.join('project', 'training_api', 'user_readapted_trainparams.json')  

        with open(trainparams_json, 'r') as data_file:
            model_data = json.load(data_file)

        # SET/CREATE NET NAMEvoicebank_vocabulary_path & PATH
        data_path = os.path.join(session_path, 'data')
        sModelFileName = model_data['sModelFileName'] 
        output_net_name = "%s_%s_%s" % (sModelFileName, str(modeltype), str_proc_scheme)
        output_net_path = os.path.join(data_path, 'net')
        os.makedirs(output_net_path)

        # CONTEXTING DATA (create ctx_...  files)
        ctx_frames = model_data['nContextFrames']   
        if ctx_frames > 0:
            context.createSubjectContext(data_path, ctx_frames)
            data_matrix, label_matrix = utilities.getSubjectTrainingMatrix(data_path, commands_ids, range(0,250), 'ctx')
        else:
            data_matrix, label_matrix = utilities.getSubjectTrainingMatrix(data_path, commands_ids, range(0,250), '')

        training_data_len = len(data_matrix[0])     # colonne: input layer length
        ncommands = len(commands_ids)
        initnet_vars_name = model_data['ftnet_vars_name']

        # START TRAINING
        if modeltype == 274:    # PU
            net_input, net_output, y = trainPureUser(training_data_len, ncommands, initnet_vars_name)
            trainModel(data_matrix, label_matrix, model_data, net_input, net_output, y, output_net_name, session_path)

        elif modeltype == 275:  # PUA (retrieve the pb file of the init PU session specified)
            init_training_session = TrainingSession.query.filter_by(session_uid=str(session_data['init_sessionid'])).first()
            if init_training_session is None:
                return # TODO : raise errors to user... I 
            else:
                model_data['init_net_path'] = init_training_session.net_path

            net_input, net_output, y, graph = trainAdapted(training_data_len, ncommands, initnet_vars_name, init_training_session.net_path)  
            trainModel(data_matrix, label_matrix, model_data, net_input, net_output, y, output_net_name, session_path, graph)

        elif modeltype == 276:  # CA (retrieve the pb file of the common session specified)
            admin_id = User.query.filter_by(role="admin").first()
            input_initmodel_path = model_data['init_net_path']

            # get last trainingSession posted by ADMIN (thus a common net) with same PREPROC method
            init_training_session = TrainingSession.query.filter_by(user_id=admin_id, preproc_type=session_data['nProcessingScheme']).order_by(TrainingSession.id.desc()).first()
            if init_training_session is None:
                # TODO : raise errors to user... I 
                return
            else:
                model_data['init_net_path'] = init_training_session.net_path

            net_input, net_output, y, graph = trainAdapted(training_data_len, ncommands, initnet_vars_name, init_training_session.net_path)
            trainModel(data_matrix, label_matrix, model_data, net_input, net_output, y, output_net_name, session_path, graph)
            
        elif modeltype == 277:  # URA
            init_training_session = TrainingSession.query.filter_by(session_uid=str(session_data['init_sessionid'])).first()
            if init_training_session is None:
                # TODO : raise errors to user... I 
                return
            else:
                model_data['init_net_path'] = init_training_session.net_path

            net_input, net_output, y, graph = trainReAdapted(training_data_len, ncommands, initnet_vars_name, init_training_session.net_path)
            trainModel(data_matrix, label_matrix, model_data, net_input, net_output, y, output_net_name, session_path, graph)


        print(voicebank_vocabulary_path)
        utilities.createVocabularyJson(commands_ids, model_data, session_data, training_sessionid, voicebank_vocabulary_path, os.path.join(session_path, 'vocabulary.json'))

        training_session = TrainingSession.query.filter_by(session_uid=str(training_sessionid)).first()
        training_session.completed = True
        training_session.net_path = os.path.join(session_path, 'optimized_' + output_net_name + '.pb')
        db.session.add(training_session)
        db.session.commit()
        print('training completed')

    except Exception as e:
        print(str(e))        
# ===========================================================================================================================
def trainPureUser(training_data_len, ncommands, initnet_vars_name):

    with tf.name_scope("inputs"):
        x = tf.placeholder(tf.float32, [None, training_data_len], name='I')
        y = tf.placeholder(tf.float32, [None, ncommands], name='y-input')

    net_inout = models.new_ff_model3(x, training_data_len, ncommands, initnet_vars_name)
    return net_inout["input"], net_inout["output"], y

def trainAdapted(training_data_len, ncommands, initnet_vars_name, input_initmodel_path):

    with tf.name_scope("inputs"):
        x = tf.placeholder(tf.float32, [None, training_data_len], name='I')
        y = tf.placeholder(tf.float32, [None, ncommands], name='y-input')

    graph = freeze.load(input_initmodel_path)
    
    net_inout = models.adapt_ff_model3(x, training_data_len, ncommands, initnet_vars_name, graph)
    return net_inout["input"], net_inout["output"], y, graph

def trainReAdapted(training_data_len, ncommands, initnet_vars_name, input_initmodel_path):

    # GET MODEL DATA
    initnet_vars_name = model_data['ftnet_vars_name']
    input_initmodel_path = model_data['init_net_path'] #

    with tf.name_scope("inputs"):
        x = tf.placeholder(tf.float32, [None, training_data_len], name='I')
        y = tf.placeholder(tf.float32, [None, ncommands], name='y-input')

    graph = freeze.load(input_initmodel_path)

    net_inout = models.readapt_ff_adaptedmodel(x, training_data_len, ncommands, initnet_vars_name, graph)
    return net_inout["input"], net_inout["output"], y, graph

# ===========================================================================================================================

def trainModel(train_data_matrix, train_label_matrix, model_data, input_layer, out_layer, y, output_model_name, session_path, graph=None, clean_folder=False):

    # temp folder to create NET files
    train_output_net_path = os.path.join(session_path, 'data', 'net')  # instance/users_data/APIKEY/train_data/training_sessionid/data/net  

    initnet_vars_name = model_data['ftnet_vars_name']
    batch_size = model_data['batch_size']
    hm_epochs = model_data['hm_epochs']
    sOutputNodeName = model_data['sOutputNodeName'] #

    # 'Saver' op to save and restore first 3 layers
    saver = tf.train.Saver()

    # Running first session
    print("Starting 1st session...")
    if graph is None:
        with tf.Session() as sess:
            with tf.device('/cpu:0'):
                start_time = time.time()
                train(sess, saver, train_data_matrix, train_label_matrix, initnet_vars_name, input_layer, out_layer, 
                    y, train_output_net_path, output_model_name, hm_epochs, batch_size)
                elapsed_time = time.time() - start_time
                print("elapsed time : " + str(elapsed_time))

            with tf.device('/cpu:0'):
                result = freeze.freeze(output_model_name, train_output_net_path, sOutputNodeName)

    else:
        with tf.Session(graph=graph) as sess:
            with tf.device('/cpu:0'):
                start_time = time.time()
                train(sess, saver, train_data_matrix, train_label_matrix, initnet_vars_name, input_layer, out_layer, 
                    y, train_output_net_path, output_model_name, hm_epochs, batch_size)
                elapsed_time = time.time() - start_time
                print("elapsed time : " + str(elapsed_time))

            with tf.device('/cpu:0'):
                result = freeze.freeze(output_model_name, train_output_net_path, sOutputNodeName)

    # move optimized net to session path
    os.rename(result['grp_opt_name'], os.path.join(session_path, 'optimized_' + output_model_name + '.pb'))
    # delete other files
    if clean_folder is True and os.path.isdir(os.path.join(session_path, 'data')) is True:
        shutil.rmtree(os.path.join(session_path, 'data'))
        

# ===========================================================================================================================
# ===========================================================================================================================
# TRAIN PRIMITIVES
# ===========================================================================================================================
# ===========================================================================================================================

# ===========================================================================================================================
# trainPureUser(train_data_matrix, train_label_matrix, commands_list, output_model_name,
#               train_output_net_path, voicebank_vocabulary_path,
#               model_type_json_path, sessiondata, clean_folder=False):
# aims      :   get train data/label matrices and create a PURE USER NEU
#               This script define the model, prepares it, calls the train process and finally create the net json
# input     :
#               train_data_matrix              training matrix with cepstra
#               train_label_matrix             training matrix with labels
#               commands_list                  list of commands ids
#               output_model_name              name of the net file : (sModelFileName, str(modeltype), str_proc_scheme)
#               train_output_net_path          full path to output net  output/train/$output_model_name
#               voicebank_vocabulary_path      path 2 file containing the global list of commands (commands list is a subset of it)
#               model_type_json_path           path to model params: {"nInputParams":792,"nContextFrames":5,"sModelFileName":"default",
#                                                                     "sInputNodeName":"inputs/I","sOutputNodeName":"SMO","fRecognitionThreshold":0.1,
#                                                                     "batch_size":500,"hm_epochs":8,"ftnet_vars_name":"model"
#               sessiondata                    {'nProcessingScheme': processing_scheme, 'sLabel': "default", 'sLocalFolder': "default", 'nModelType': 273
#               clean_folder=False
# ===========================================================================================================================

# ===========================================================================================================================



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
def train(sess, saver, training_data, training_label, train_vars_name, input_layer, output_layer, y, output_dir, net_name, hm_epochs=20, batch_size=100):

    # COST FUNCTION:
    cost_ap = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_layer, labels=y))
    train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, train_vars_name)
    optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost_ap, var_list=train_vars)

    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    # save the graph
    # tf.train.write_graph(sess.graph_def, output_dir, "graph.pb", False)
    tf.train.write_graph(sess.graph_def, output_dir, net_name + '.pbtxt')

    nframes = len(training_data)
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

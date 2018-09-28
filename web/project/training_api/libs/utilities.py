# createSubjectTrainingMatrix(subj, in_orig_subj_path, output_net_path, arr_commands, arr_rip)
# createSubjectTestMatrix(subj, in_orig_subj_path, output_net_path, arr_commands, arr_rip, sentences_filename, sentence_counter)
# createFullMatrix(input_matrix_folder, data_name, label_name, output_matrix_path="")

import sys
import os
import shutil
import ntpath
import glob
import re
import json
import numpy as np
from numpy import genfromtxt
from datetime import datetime
from . import earray_wrapper
from __future__ import print_function
import tensorflow as tf


def moveFolderContent(indir, outdir):
    for file in os.listdir(indir):
        fileref = indir + '/' + file
        if os.path.isdir(fileref) is False:
            shutil.move(fileref, outdir + '/' + file)


def getFileName(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


def remoteExtension(path):
    return path.split('.')[0]


# works when input files are : commandlabelNREP_scores.dat
# vocfilepath contains lines as follows:
# 1 cmdlab1
# 2 cmdlab7
# 3 cmlab8
# etc...
def renameSubjectFiles(subject_name, inrootpath, outrootpath, vocfilepath):
    inpath = inrootpath + '/' + subject_name
    outpath = outrootpath + '/' + subject_name
    if os.path.isdir(outpath) is False:
        os.mkdir(outpath)

    with open(vocfilepath, "r") as f:
        data = f.readlines()
        for line in data:
            words = line.split()  # words[0]=num corrispondente al comando, words[1]=nome frase
            for infile in glob.glob(os.path.join(inpath, '*.*')):
                file_name = os.path.basename(infile)
                b = re.split('(\d+)', file_name)
                if b[0] == words[1]:
                    shutil.copy2(infile, outpath + '/' + subject_name + "_" + words[0] + "_" + b[1] + ".dat")
                    # print subject_name + "_" + words[0] + "_" + b[1] + ".dat" -> te_0_0.dat


# works when input files are : SUBJ_commandlabelNREP_scores.dat
# vocfilepath contains lines as follows:
# 1 cmdlab1
# 2 cmdlab7
# 3 cmlab8
# etc...
def renameSubjectsFiles(subjects_name, inrootpath, outrootpath, vocfilepath):
    inpath = inrootpath + '/' + subjects_name

    with open(vocfilepath, "r") as f:
        data = f.readlines()
        for line in data:
            words = line.split()  # words[0]=num corrispondente al comando, words[1]=nome frase
            for infile in glob.glob(os.path.join(inpath, '*.*')):
                file_name = os.path.basename(infile)
                id = file_name.index("_")
                subjlabel = file_name[:id]

                if os.path.isdir(outrootpath + '/' + subjlabel) is False:
                    os.mkdir(outrootpath + '/' + subjlabel)

                file_name = file_name[(id+1):]

                b = re.split('(\d+)', file_name)
                if b[0] == words[1]:
                    shutil.copy2(infile,
                                 outrootpath + '/' + subjlabel + '/' + subjlabel + "_" + words[0] + "_" + b[1] + ".dat")
                    # print subject_name + "_" + words[0] + "_" + b[1] + ".dat" -> te_0_0.dat


# works when input files are : commandlabelNREP_scores.dat
# vocfilepath contains lines as follows:
# 1 cmdlab1
# 2 cmdlab7
# 3 cmlab8
# etc...
def renameSubjectFilesJSON(subject_name, inrootpath, outrootpath, jsonvocfilepath, ext=".dat"):
    inpath = inrootpath + '/' + subject_name
    outpath = outrootpath + '/' + subject_name
    if os.path.isdir(outpath) is False:
        os.mkdir(outpath)

    vocabulary = getVocabularyFromJSON(jsonvocfilepath)

    for sentence in vocabulary:
        sentenceid = str(sentence["id"])
        lab = remoteExtension(str(sentence["readablefilename"]))

    #with open(vocfilepath, "r") as f:
    #    data = f.readlines()
    #    for line in data:
    #        words = line.split()  # words[0]=num corrispondente al comando, words[1]=nome frase
        for infile in glob.glob(os.path.join(inpath, '*.*')):
            file_name = os.path.basename(infile)
            b = re.split('(\d+)', file_name)
            if b[0] == lab:
                shutil.copy2(infile, outpath + '/' + subject_name + "_" + sentenceid + "_" + b[1] + ext)
                # print subject_name + "_" + words[0] + "_" + b[1] + ".dat" -> te_0_0.dat


# works when input files are : SUBJ_commandlabelNREP_scores.dat
# jsonvocfilepath contains lines as follows:
#{    "vocabulary_categories": [],
#    "voicebank_vocabulary": [ { "title": "Sono felice", "id": 1101, "filename":"", "readablefilename" : "sono_felice.wav", "existwav": 0, "editable":false}, ...]
#}
def renameSubjectsFilesJSON(subjects_name, inrootpath, outrootpath, jsonvocfilepath, ext=".dat"):
    inpath = inrootpath + '/' + subjects_name

    vocabulary = getVocabularyFromJSON(jsonvocfilepath)


    # words = line.split()  # words[0]=num corrispondente al comando, words[1]=nome frase
    for infile in glob.glob(os.path.join(inpath, '*.*')):
        copied = False
        file_name = os.path.basename(infile)
        id = file_name.index("_")
        subjlabel = file_name[:id]

        if os.path.isdir(outrootpath + '/' + subjlabel) is False:
            os.mkdir(outrootpath + '/' + subjlabel)

        file_name = file_name[(id + 1):]
        b = re.split('(\d+)', file_name)
        for sentence in vocabulary:
            sentenceid = str(sentence["id"])
            lab = remoteExtension(str(sentence["readablefilename"]))
            if b[0] == lab:

                if os.path.isdir(outrootpath + '/' + subjlabel) is False:
                    os.mkdir(outrootpath + '/' + subjlabel)

                shutil.copy2(infile, outrootpath + '/' + subjlabel + '/' + subjlabel + "_" + sentenceid + "_" + b[1] + ext)
                shutil.copy2(infile, outrootpath + '/' + subjlabel + "_" + sentenceid + "_" + b[1] + ext)
                copied = True
                break
                # print subject_name + "_" + words[0] + "_" + b[1] + ".dat" -> te_0_0.dat
        if copied is False:
            print(infile)



        # works when input files are : commandlabelNREP.dat.SUBJLABEL


def renameSubjectFilesOld(subject_name, inrootpath, outrootpath, vocfilepath):
    inpath = inrootpath + '/' + subject_name
    outpath = outrootpath + '/' + subject_name
    if os.path.isdir(outpath) is False:
        os.mkdir(outpath)

    with open(vocfilepath, "r") as f:
        data = f.readlines()
        for line in data:
            words = line.split()  # words[0]=num corrispondente al comando, words[1]=nome frase
            for infile in glob.glob(os.path.join(inpath, '*.*')):
                file_name = os.path.basename(infile)
                a = os.path.splitext(file_name)[0]
                b = re.split('(\d+)', a)
                if b[0] == words[1]:
                    shutil.copy2(inpath + '/' + a + '.' + subject_name,
                                 outpath + '/' + subject_name + "_" + words[0] + "_" + b[1] + ".dat")
                    # print subject_name + "_" + words[0] + "_" + b[1] + ".dat" -> te_0_0.dat


def getVocabularyFromJSON(json_inputfile):
    with open(json_inputfile, encoding='utf-8') as data_file:
        data = json.load(data_file)
    return data["voicebank_vocabulary"]


def createVocabularySentence(list_ids, json_inputfile, txt_outputfile):
    vocabulary = getVocabularyFromJSON(json_inputfile)
    file = open(txt_outputfile, 'w+')
    for id in list_ids:
        for sentence in vocabulary:
            sentenceid = sentence["id"]
            if id == sentenceid:
                title = sentence["title"]
                file.write(title + os.linesep)
                break
    file.close()


def createVocabularyJson(list_ids, model, sessiondata, training_sessionid, json_globalvocabulary, json_outputfile):

    # get commands list from json_globalvocabulary
    vocabulary = getVocabularyFromJSON(json_globalvocabulary)
    commands = []
    for id in list_ids:
        for sentence in vocabulary:
            sentenceid = sentence["id"]
            if id == sentenceid:
                commands.append({'title': sentence["title"], 'id': sentenceid})
                break
    lencmds = len(commands)

    nw = datetime.now()

    # sModelFilePath is written by the App
    res = {
           'sLabel': sessiondata['sLabel'],
           'nModelClass': sessiondata['nModelClass'],
           'nModelType': sessiondata['nModelType'],
           'nInputParams': model['nInputParams'],
           'nContextFrames': model['nContextFrames'],
           'nItems2Recognize': lencmds,
           'sModelFilePath': "",
           'sModelFileName': model['sModelFileName'],
           'saInputNodeName': model['saInputNodeName'],
           'sOutputNodeName': model['sOutputNodeName'],
           'nProcessingScheme': sessiondata['nProcessingScheme'],
           'fRecognitionThreshold': model['fRecognitionThreshold'],
           'sCreationTime': nw.strftime('%Y/%m/%d %H:%M:%S'),
           'sLocalFolder': sessiondata['sLocalFolder'],
           'sessionid': str(training_sessionid),
           'commands': commands
           }

    with open(json_outputfile, 'w', encoding='utf-8') as data_file:
        json.dump(res, data_file)


# ===========================================================================================================================
# aims      :   This script creates the training matrix for a single subject (ctx_*.dat ==> SUBJ_train_data.npy [earray h5])
#
# input     :   subj: subject folder name
#               in_orig_subj_path: path to the subject's cepstra with context (instance/users_data/APIKEY/train_data/training_sessionid/data)
#               output_net_path: path to the output folder
#               arr_commands: IDs of the selected commands
#               arr_rip: range from 0 to Nripetitions
#
# return    :   output_matrices_path: path to the output folder (e.g.  output/train/ANALYSISNAME/matrices)
# ===========================================================================================================================
def createSubjectTrainingMatrix(subj, in_orig_subj_path, output_net_path, arr_commands, arr_rip, file_prefix='ctx'):
    mat_compl = []
    mat_lab = []

    totalsize = 0
    output_matrices_path = os.path.join(output_net_path, 'matrices')

    write_every_nfiles = 1     # every N (e.g. 10) files read, append them to disk and clear arrays

    if os.path.isdir(output_matrices_path) is False:
        os.mkdir(output_matrices_path)

    if subj != '':
        subj = subj + "_"

    output_data_matrix_path = output_matrices_path + '/' + subj + 'train_data.npy'
    output_labels_matrix_path = output_matrices_path + '/' + subj + 'train_labels.npy'

    if os.path.exists(output_data_matrix_path) is True:
        os.remove(output_data_matrix_path)

    if os.path.exists(output_labels_matrix_path) is True:
        os.remove(output_labels_matrix_path)

    try:
        cnt = 0
        for ctxfile in glob.glob(in_orig_subj_path + '/' + file_prefix + '*'):

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
                cnt = cnt + 1

                # check whether write 2 disk
                if cnt == write_every_nfiles:
                    cnt = 0
                    earray_wrapper.appendArray2File(mat_compl, output_data_matrix_path)
                    earray_wrapper.appendArray2File(mat_lab, output_labels_matrix_path)
                    totalsize += mat_compl.size
                    mat_compl = []
                    mat_lab = []

    except Exception as e:
        print(str(e))

    # save data in output/train/ANALYSISNAME/matrices
    if len(mat_compl):
        earray_wrapper.appendArray2File(mat_compl, output_data_matrix_path)
        earray_wrapper.appendArray2File(mat_lab, output_labels_matrix_path)
    print("createSubjectTrainingMatrix ended: " + str(totalsize))

    return {'data_matrices_path': output_data_matrix_path, 'labels_matrices_path': output_labels_matrix_path}

# -----------------------------------------------------------------------------------------------------------------------
# DO NOT create matrices file, just read and returns the data & labels arrays
def getSubjectTrainingMatrixFF(in_orig_subj_path, arr_commands, arr_rip, file_prefix='ctx'):

    mat_compl = []
    mat_lab = []
    totalsize = 0
    try:
        cnt = 0
        for ctxfile in glob.glob(in_orig_subj_path + '/' + file_prefix + '*'):

            filename = ctxfile.split('/')[-1]
            spl = filename.split('.')[0]
            spl = spl.split('_')
            # spl = re.split('[_ .]', filename)  # e.g. ctx_SUBJ_CMD_REP => spl2[2] num comando, spl[3] num ripetiz
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
                cnt = cnt + 1


    except Exception as e:
        print(str(e))

    rows = len(mat_compl)
    cols = len(mat_compl[0])
    print("getSubjectTrainingMatrix ended, row: " + str(rows)+ ", cols: " + str(cols))

    return mat_compl, mat_lab
    # return {'data_matrices': mat_compl, 'labels_matrices': mat_lab}


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

    totalsize = 0
    output_matrices_path = os.path.join(output_net_path, 'matrices')

    write_every_nfiles = 10     # every N (e.g. 10) files read, append them to disk and clear arrays

    if os.path.isdir(output_matrices_path) is False:
        os.mkdir(output_matrices_path)

    if os.path.isfile(sentences_filename) is True:
        os.remove(sentences_filename)

    output_data_matrix = output_matrices_path + '/' + subj + '_test_data.npy'
    output_labels_matrix = output_matrices_path + '/' + subj + '_test_labels.npy'

    if os.path.exists(output_data_matrix) is True:
        os.remove(output_data_matrix)

    if os.path.exists(output_labels_matrix) is True:
        os.remove(output_labels_matrix)

    try:
        cnt = 0
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
                cnt = cnt + 1

                # check whether write 2 disk
                if cnt == write_every_nfiles:
                    cnt = 0
                    earray_wrapper.appendArray2File(mat_compl, output_data_matrix)
                    earray_wrapper.appendArray2File(mat_lab, output_labels_matrix)
                    totalsize += mat_compl.size
                    mat_compl = []
                    mat_lab = []

    except Exception as e:
        print(str(e))

    # save data in output/test/ANALYSISNAME/matrices
    if len(mat_compl):
        earray_wrapper.appendArray2File(mat_compl, output_data_matrix)
        earray_wrapper.appendArray2File(mat_lab, output_labels_matrix)

    return {'data_matrices_path': output_data_matrix, 'labels_matrices_path': output_labels_matrix, 'sentence_counter': sentence_counter}


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
def createFullMatrix(subjects_list, input_net_folder, data_name, label_name, output_net_folder=""):

    input_matrix_folder = os.path.join(input_net_folder, 'matrices')
    if os.path.isdir(input_matrix_folder) is False:
        os.mkdir(input_matrix_folder)

    if len(output_net_folder):
        output_matrix_folder = os.path.join(output_net_folder, 'matrices')
        if os.path.isdir(output_matrix_folder) is False:
            os.mkdir(output_matrix_folder)

        data_matrix_path = output_matrix_folder + '/full_' + data_name + '.npy'
        label_matrix_path = output_matrix_folder + '/full_' + label_name + '.npy'
    else:
        data_matrix_path = input_matrix_folder + '/full_' + data_name + '.npy'
        label_matrix_path = input_matrix_folder + '/full_' + label_name + '.npy'

    for file in glob.glob(input_matrix_folder + '/*' + data_name + '.npy'):
        file_name = os.path.basename(file)

        spl = re.split('[_ .]', file_name)  # spl[0] paz, spl[1] parola 'train'

        for subj in subjects_list:
            if subj == spl[0]:
                print("createFullMatrix: " + file)

                file_train = np.load(file)
                file_labels = np.load(input_matrix_folder + '/' + spl[0] + '_' + label_name + '.npy')

                earray_wrapper.appendArray2File(file_train, data_matrix_path)
                earray_wrapper.appendArray2File(file_labels, label_matrix_path)

    return {'data_matrix_path': data_matrix_path, 'label_matrix_path': label_matrix_path}


def getNodeBySubstring(graph, nomesubstring, allnodes=None):
    if allnodes is None:
        allnodes = [n.name for n in graph.as_graph_def().node ]

    node_str = [s for s in allnodes if nomesubstring in s and 'read' not in s]
    if len(node_str) == 1:
        return graph.get_tensor_by_name(node_str[0] + ":0")
    else:
        return None


# ===========================================================================================================================
# LSTM Auxiliary functions
# ===========================================================================================================================

# ===========================================================================================================================
# read *dat from a given folder and create corresponding TFRecords file
# in_orig_subj_path         (instance/users_data/APIKEY/train_data/training_sessionid/data)
def createSubjectTrainingTFRecords(in_orig_subj_path, arr_commands):

    train_filenames = [name for name in glob.glob(os.path.join(in_orig_subj_path,'*.dat'))]

    # Build tfrecords
    #filelist = open('training_files_list.txt','w')
    for index,file in enumerate(train_filenames):

        #print('serializing TRAIN file {} of {}'.format(index,len(filelist)))
        #filelist.write(file+'\t'+str(index)+'\n')
        features = np.genfromtxt(file, dtype=float)

        for key,value in arr_commands.iteritems():
            if key in file:
                label = float(value)

        filename            = os.path.join(in_orig_subj_path, 'sequence_full_{:04d}.tfrecords'.format(index))
        fp                  = open(filename,'w')
        writer              = tf.python_io.TFRecordWriter(fp.name)
        serialized_sentence = serialize_sequence(features, label)

        # write to tfrecord
        writer.write(serialized_sentence.SerializeToString())
        writer.close()
        fp.close()
    #filelist.close()
    return len(train_filenames)

# accessory function for the above createSubjectTrainingTFRecords
def serialize_sequence(audio_sequence, label):
    # The object we return
    ex = tf.train.SequenceExample()
    # A non-sequential feature of our example
    sequence_length = len(audio_sequence)
    ex.context.feature["length"].int64_list.value.append(sequence_length)

    # Feature lists for the two sequential features of our example
    fl_audio_feat = ex.feature_lists.feature_list["audio_feat"]
    fl_audio_labels = ex.feature_lists.feature_list["audio_labels"]
    fl_audio_labels.feature.add().float_list.value.append(label)

    for audio_feat in audio_sequence:
        fl_audio_feat.feature.add().float_list.value.extend(audio_feat)    

    return ex

# ===========================================================================================================================
# create input pipeline
# ===========================================================================================================================
def input_pipeline(filenames, model_data):

    filename_queue = tf.train.string_input_producer(filenames, num_epochs=model_data.num_epochs, shuffle=True)
    
    sequence_length, audio_features, audio_labels = read_my_file_format(filename_queue,feat_dimension=model_data.nInputParams)

    audio_features_batch, audio_labels_batch , seq_length_batch = tf.train.batch([audio_features, audio_labels, sequence_length],
                                                    batch_size  = model_data.batch_size,
                                                    num_threads = 10,
                                                    capacity    = 100,
                                                    dynamic_pad = True,
                                                    enqueue_many= False)

    return audio_features_batch, audio_labels_batch, seq_length_batch

# Reads a single serialized SequenceExample
def read_my_file_format(filename_queue,feat_dimension=72):

    reader = tf.TFRecordReader()

    key, serialized_example = reader.read(filename_queue)

    context_parsed, sequence_parsed = tf.parse_single_sequence_example(serialized_example,
                                               context_features={"length": tf.FixedLenFeature([],dtype=tf.int64)},
                                               sequence_features={"audio_feat":tf.FixedLenSequenceFeature([feat_dimension],dtype=tf.float32),
                                                                  "audio_labels":tf.FixedLenSequenceFeature([],dtype=tf.float32)}
                                        )

    return context_parsed['length'],sequence_parsed['audio_feat'],tf.to_int32(sequence_parsed['audio_labels'])

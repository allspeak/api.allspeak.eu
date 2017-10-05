# This module is related to the test phase and it contains all the functions listed below:
#
# classify(sess, input_layer, out_layer_sm, test_data, out_res_file)
# test(sess, input_layer, out_layer_sm, ncommands, sentence_counter_file, out_res_file, output_NR_file, test_data, test_label, threshold=0)
# Sentence(list)
# test_ft(subj, sentence_counter_filename, test_data_filename, test_label_filename, test_data_root_path, output_net_root_path, output_model_name_train, output_model_name_test, output_res_filename, output_NR_filename, ncommands, thr)

import tensorflow as tf
import numpy as np
from numpy import genfromtxt
from operator import add
import operator
import os
from . import freeze


#===========================================================================================================================
# aims      :   This script calculates the neural network accuracy based on the number of correct predictions
#
# input     :   sess: session currently running
#               input_layer: layer named as "I" in 2_main_train_net.py
#               out_layer_sm: softmax layer
#               ncommands: commands' number
#               sentence_counter_file: it indicates to which sentence each frame belong
#               out_res_file: path to the output file which shows the test results
#               output_NR_file: path to the output file containing the command_id of the sentences not recognized
#               test_data: testing matrix with cepstra
#               test_label: testing matrix with labels
#               threshold: threshold value. It should be: 0.0, 0.10, 0.15, 0.20
#
# return    :   final_sentences_prob: final probabilities for each command
#===========================================================================================================================

# def test(sess, input_layer, out_layer_sm, ncommands, sentence_counter_file, out_res_file, output_NR_file, test_data, test_label, threshold=0):
def test(sess, input_layer, out_layer_sm, ncommands, test_data, test_label, prob_out_file_path=""):

    with tf.name_scope("inputs"):
        y = tf.placeholder(tf.float32, [None, ncommands], name='y-input')

    # Calculate accuracy
    correct = tf.equal(tf.argmax(out_layer_sm, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, "float"), name="accuracy")
    print('Accuracy:', sess.run(accuracy, feed_dict={input_layer: test_data, y: test_label}))

    probabilities = sess.run(out_layer_sm, feed_dict={input_layer: test_data, y: test_label})

    if(len(prob_out_file_path)):
        np.savetxt(prob_out_file_path, probabilities, "%.4f")
    return probabilities


def getSentencesProbabilitiesFromFile(prob_path, sentence_counter_file, out_res_file, output_NR_file, threshold=0):
    getSentencesProbabilities(genfromtxt(prob_path), sentence_counter_file, out_res_file, output_NR_file, threshold)


def getSentencesProbabilities(probabilities, sentence_counter_file, commands_list, out_res_file, output_NR_file, threshold=0):

    nframes = len(probabilities)
    ncommands = len(probabilities[0])

    arr_sequences = genfromtxt(sentence_counter_file)
    nseqrows = len(arr_sequences)

    if nseqrows != nframes:
        print("ERROR in getSentencesProbabilities : output probabilites number does not coincide with model sentences !! ....exiting")
        return 0

    final_sentences_prob = []
    curr_sentence_prob = [0 for _ in range(ncommands)]
    curr_seq_id = -1
    curr_cmd_id = -1
    curr_net_id = -1
    sentence_frames = 0
    cnt = 0

    for row in arr_sequences:
        id_seq = int(row[0])
        id_cmd = int(row[1])    # file id : eg. 1401

        id_net = -1             # order-id within the net file : eg. 2
        cn = 0
        for cmdlabel in commands_list:
            if cmdlabel == id_cmd:
                id_net = cn
                break
            cn = cn + 1



        frame_prob = probabilities[cnt] # list of ncommands float

        if id_seq == curr_seq_id or cnt == 0:       # first frame o continue current sentence
            if np.any(frame_prob >= threshold):
                curr_sentence_prob = map(add, curr_sentence_prob, frame_prob)
                sentence_frames = sentence_frames + 1

        else:   # new sequence

            # store previous sentence
            curr_sentence_prob = [i / sentence_frames for i in curr_sentence_prob]
            final_sentences_prob.append(Sentence(curr_seq_id, curr_cmd_id, curr_net_id, curr_sentence_prob))

            if np.any(frame_prob >= threshold):
                # process first frame of the new sequence
                curr_sentence_prob = frame_prob
                sentence_frames = 1

            else:
                curr_sentence_prob = [0 for _ in range(ncommands)]
                sentence_frames = 1

        cnt = cnt + 1
        curr_seq_id = id_seq
        curr_cmd_id = id_cmd
        curr_net_id = id_net

    # add last frame to the last sentence
    curr_sentence_prob = [i / sentence_frames for i in curr_sentence_prob]
    final_sentences_prob.append(Sentence(id_seq, id_cmd, id_net, curr_sentence_prob))

    # parse final_sentences_prob
    cnt=0
    res=""
    success_rate=0.0
    for sentence in final_sentences_prob:
        res = res + sentence.toString(ncommands)
        if sentence.isCorrect == 0:
            if cnt == 0:
                with open(output_NR_file, "w") as txt_file:
                    txt_file.write(str(sentence.command_id) + ", ")
            else:
                with open(output_NR_file, "a") as txt_file:
                    txt_file.write(str(sentence.command_id) + ", ")
        success_rate = success_rate + sentence.isCorrect
        cnt = cnt + 1

    success_rate = (success_rate/cnt)*100

    # write res 2 text
    with open(out_res_file, "w") as text_file:
        text_file.write("Success Rate = " + str("%.1f" % success_rate)+"\n")
        text_file.write("sID\tcmdID\tnetID\tisCorr\tRecID\tProb\n")
        text_file.write(res)

    return final_sentences_prob

#===========================================================================================================================
# aims      :   This script restores the .pb file after the fine tuning training and calculates the neural network accuracy
#
# input     :   subj: subject name
#               sentence_counter_filename: it takes account of how many rows are occupied by each command and the command_id
#               test_data_filename: testing matrix with cepstra coefficients
#               test_label_filename: testing matrix with labels
#               test_data_root_path: general path to all the testing folders
#               output_net_root_path: general path to all the pretraining nets
#               output_model_name_train: folder named as subj_outnet_name in 3_main_fine_tuning.py
#               output_model_name_test: folder named as subj_outnet_name in 3_main_fine_tuning.py
#               output_res_filename: path to the output file which shows the test results
#               output_NR_filename: path to the output file containing the command_id of the sentences not recognized
#               ncommands: number of commands
#               thr: threshold values. It should be: 0.0, 0.10, 0.15, 0.20
#
# return    :   void
#===========================================================================================================================

# def test_ft(subj, sentence_counter_filename, test_data_filename, test_label_filename, test_data_root_path, output_net_root_path, output_model_name_train, output_model_name_test, output_res_filename, output_NR_filename, ncommands, thr, prob_out_file_path=""):
def test_ft(subj, test_data_filename, test_label_filename, test_data_root_path, output_net_root_path,
            output_model_name_train, output_model_name_test, ncommands, prob_out_file_path=""):

    # def path
    input_model_path = output_net_root_path + "/" + output_model_name_train + "/optimized_" + output_model_name_train + ".pb"

    input_data_path = test_data_root_path + "/" + output_model_name_test + "/matrices/" + subj + test_data_filename
    input_labels_path = test_data_root_path + "/" + output_model_name_test + "/matrices/" + subj + test_label_filename

    # load data
    test_data = genfromtxt(input_data_path)
    test_labels = genfromtxt(input_labels_path)

    # restore PB
    graph = freeze.load(input_model_path)

    with tf.Session(graph=graph) as sess:
        input_layer = graph.get_tensor_by_name('prefix/inputs/I:0')
        output_sm_layer = graph.get_tensor_by_name('prefix/SMO:0')

        return test(sess, input_layer, output_sm_layer, ncommands, test_data, test_labels, prob_out_file_path)


# ===========================================================================================================================
# aims      :   This class returns a text file listing the neural network accuracy based on the number of correct predictions,
#               specifying which commands have been rightly predicted or not (with related probability values)
#
# methods   :   getMostProbableID: it returns the number of the command predicted by the neural network looking for the maximum
#               value and position among the probabilities
#
#               toString(.., ncommands): it fills in the output file with information related to the test phase (accuracy, recognized
#               commands, and so on)
#
# return    :   void
# ===========================================================================================================================
class Sentence(list):
    def __init__(self, id_sentence, id_cmd, id_net, probs):
        self.id = id_sentence
        self.command_id = id_cmd    # 4-digits number  eg. 1401
        self.netorder_id = id_net   # 0-based id
        self.probabilities_list = [round(elem, 4) for elem in probs]
        self.recognized = self.getMostProbableID()  # 0-based id
        if self.recognized == self.netorder_id:
            self.isCorrect = 1
        else:
            self.isCorrect = 0

    def getMostProbableID(self):
        index, value = max(enumerate(self.probabilities_list), key=operator.itemgetter(1))
        return index

    def toString(self, ncommands):
        res = ""
        res = str(self.id) + "\t" + str(self.command_id) + "\t" + str(self.netorder_id) + "\t" + str(self.isCorrect) + "\t" + str(self.recognized) + "\t"
        for c in range(ncommands):
            res = res + str(self.probabilities_list[c]) + ", "
        res = res + "\n"

        return res


#===========================================================================================================================
# aims      :   This script returns the index of the max value for each row of the resulting probabilities during the test phase
#
# input     :   sess: session currently running
#               input_layer: layer named as "I" in  4_main_classify_single_sentence.py
#               out_layer_sm: softmax layer
#               test_data: testing matrix with cepstra
#               out_res_file: path to the output file which shows the test results
#
# return    :   index: neural network prediction, a number ranging from [1 to Ncommands]
#===========================================================================================================================
def classify(sess, input_layer, out_layer_sm, test_data, out_res_file):
    probabilities = sess.run(out_layer_sm, feed_dict={input_layer: test_data})
    out_res_file = np.savetxt(out_res_file,probabilities, fmt='%.4f')

    ncommands = len(probabilities[0])
    sentence_prob = [0 for _ in range(ncommands)]
    nfr=0
    for row in probabilities:
        sentence_prob = map(add, sentence_prob, row)
        nfr += 1
    sentence_prob = [i / nfr for i in sentence_prob]
    index, value = max(enumerate(sentence_prob), key=operator.itemgetter(1))

    return index+1

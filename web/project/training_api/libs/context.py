#===========================================================================================================================
# aims      :   This script takes the frame file in input and gives it back adding the context to each frame
#
# input     :   input_filepath : folder containing subjects' cepstra. It takes the subject_path set in 1_main_process_subjects.py
#
#               output_filepath: folder where it writes back the cepstra with context
#               Nframes: number of context frames added to both sides of each single frame. It takes the ctx_frames set in 1_main_process_subjects.py
#
# return    :   void
#===========================================================================================================================

import os
import re

import numpy as np
from numpy import genfromtxt
from . import utilities


def create_context_file(input_filepath, output_filepath, Nframes):

    f = open(input_filepath, 'rb')
    lines = f.readlines()
    count_lines = len(lines)  # str(len(lines))

    ctx_file = open(output_filepath, "wb")

    # TEST DATASET
    startSent = 0  # start line command
    endSent = int(count_lines) -1  # end line command
    data = genfromtxt(input_filepath)

    Newdata = np.zeros([data.shape[0], data.shape[1] * ((2 * Nframes) + 1)])

    indexData1 = startSent
    indexData2 = endSent

    # first frame of the statement:
    Newdata[indexData1, :] = np.concatenate((np.tile(data[indexData1, :], Nframes + 1),
                                         data[indexData1 + 1:indexData1 + Nframes + 1, :].reshape(
                                             data.shape[1] * Nframes)))
    indexData1 += 1

    # second frame of the statement:
    Newdata[indexData1, :] = np.concatenate((np.tile(data[startSent, :], Nframes), data[indexData1, :],
                                         data[indexData1 + 1:indexData1 + Nframes + 1, :].reshape(
                                             data.shape[1] * Nframes)))
    indexData1 += 1

    # from 3th frame to the Nframesth frame:
    while indexData1 < Nframes:
        diff1 = indexData1
        DummyFramesL = np.concatenate((np.tile(data[startSent, :], Nframes - diff1 + 1),
                                   data[indexData1 - diff1 + 1:indexData1, :].reshape(data.shape[1] * (diff1 - 1))))

        Newdata[indexData1, :] = np.concatenate((DummyFramesL, data[indexData1, :],
                                             data[indexData1 + 1:indexData1 + Nframes + 1, :].reshape(
                                                 data.shape[1] * Nframes)))
        indexData1 += 1

    # central frames :
    for index in range(indexData1, indexData2 - Nframes):
        Newdata[index, :] = data[max(0, index - Nframes - 1 + 1): index + Nframes + 1, :].reshape(
            data.shape[1] * ((2 * Nframes) + 1))

    # last frame of the statement :
    Newdata[indexData2, :] = np.concatenate((data[indexData2 - Nframes:indexData2, :].reshape(data.shape[1] * Nframes),
                                         np.tile(data[indexData2, :], Nframes + 1)))
    indexData2 -= 1

    # penultimate frame of the statement :
    Newdata[indexData2, :] = np.concatenate((data[indexData2 - Nframes:indexData2, :].reshape(data.shape[1] * Nframes),
                                         data[indexData2, :], np.tile(data[indexData2 + 1, :], Nframes)))
    indexData2 -= 1

    # frames in [-Nframes, penultimate]:
    while endSent - indexData2 < Nframes:
        diff2 = endSent - indexData2
        DummyFramesR = np.concatenate((data[indexData2 + 1:endSent, :].reshape(data.shape[1] * (diff2 - 1)),
                                    np.tile(data[endSent, :], Nframes - diff2 + 1)))

        Newdata[indexData2, :] = np.concatenate(
            (data[indexData2 - Nframes:indexData2, :].reshape(data.shape[1] * Nframes), data[indexData2, :], DummyFramesR))

        indexData2 -= 1

    np.savetxt(ctx_file, Newdata, fmt='%.4f')


def createSubjectContext(subject_path, ctx_frames):

    for f in os.listdir(subject_path):
        if f.endswith(".dat") and not f.startswith("ctx"):
            original_file_path = subject_path + "/" + f
            ctx_file_path = subject_path + "/" + 'ctx_' + f
            print(f)
            create_context_file(original_file_path, ctx_file_path, ctx_frames)
    print("done")

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

import numpy as np
from numpy import genfromtxt


def fnorm(data, m=None,sd=None):

    frames=len(data[0])
    scores=len(data[1])

    ndata = data

    if m is None:
        m = np.mean(data, 0)
        sd = np.std(data, 0)

    for col in range(scores):
        if sd[col] == 0:
            ndata[:,col] = 0
        else:
            ndata[:, col] = (data[:, col] - m[col]) / sd[col]

    # ndata = (data - m) / sd
    return(ndata, m,sd)


def normalizeSubject(subject_path):

    for file in os.listdir(subject_path):
        original_file_path = subject_path + "/" + file
        with open(original_file_path, "rb+") as f:
            contents = genfromtxt(f)
        #data = genfromtxt(original_file_path)
        (ndata,m,sd) = fnorm(contents)
        np.savetxt(original_file_path, ndata, fmt='%.4f')
    print("done")
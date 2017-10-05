import os
import shutil
import ntpath
import glob
import re
import json

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
    with open(json_inputfile) as data_file:
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


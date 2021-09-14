#  * This code is provided solely for the personal and private use of students
#  * taking the CSC401 course at the University of Toronto. Copying for purposes
#  * other than this use is expressly prohibited. All forms of distribution of
#  * this code, including but not limited to public repositories on GitHub,
#  * GitLab, Bitbucket, or any other online platform, whether as given or with
#  * any changes, are expressly prohibited.
#  
#  * All of the files in this directory and all subdirectories are:
#  * Copyright (c) 2020 Frank Rudzicz

import numpy as np
import re
import argparse
import json
import os
import csv

# Provided wordlists.
FIRST_PERSON_PRONOUNS = {
    'i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours'}
SECOND_PERSON_PRONOUNS = {
    'you', 'your', 'yours', 'u', 'ur', 'urs'}
THIRD_PERSON_PRONOUNS = {
    'he', 'him', 'his', 'she', 'her', 'hers', 'it', 'its', 'they', 'them',
    'their', 'theirs'}
SLANG = {
    'smh', 'fwb', 'lmfao', 'lmao', 'lms', 'tbh', 'rofl', 'wtf', 'bff',
    'wyd', 'lylc', 'brb', 'atm', 'imao', 'sml', 'btw', 'bw', 'imho', 'fyi',
    'ppl', 'sob', 'ttyl', 'imo', 'ltr', 'thx', 'kk', 'omg', 'omfg', 'ttys',
    'afn', 'bbs', 'cya', 'ez', 'f2f', 'gtr', 'ic', 'jk', 'k', 'ly', 'ya',
    'nm', 'np', 'plz', 'ru', 'so', 'tc', 'tmi', 'ym', 'ur', 'u', 'sol', 'fml'}

a1_path = '/u/cs401/A1'
feats_path = os.path.join(a1_path, 'feats')

file_data = {
        'Left': np.array([0]),
        'Center': np.array([1]),
        'Right': np.array([2]),
        'Alt' : np.array([3])
        }
class_data = {
        'Left': 0,
        'Center': 1,
        'Right': 2,
        'Alt': 3
        }


# Here is loading the data from /u/cs401/A1.
for fname in file_data:
    tmp = {}
    ids = open(os.path.join(feats_path, fname + '_IDs.txt')).readlines() 
    liwc = np.load(os.path.join(feats_path, fname + '_feats.dat.npy'))
    for i in range(len(ids)):
        tmp[str(ids[i]).strip()] = liwc[i]
    file_data[fname] = tmp

def return_num(x):
    if len(x) > 0:
        if x[0].isnumeric():
            return float(x)
    else:
        return np.NaN

bgl_data = {}
bgl_csv = open("/u/cs401/Wordlists/BristolNorms+GilhoolyLogie.csv")
bgl_reader = csv.DictReader(bgl_csv)
for row in bgl_reader:
    bgl_data[row["WORD"]] = [
            return_num(row["AoA (100-700)"]), 
            return_num(row["IMG"]), 
            return_num(row["FAM"])
                ]

warr_data = {}
warr_csv = open("/u/cs401/Wordlists/Ratings_Warriner_et_al.csv")
warr_reader = csv.DictReader(warr_csv)
for row in warr_reader:
    warr_data[row["Word"]] = [
            return_num(row["V.Mean.Sum"]),
            return_num(row["A.Mean.Sum"]), 
            return_num(row["D.Mean.Sum"])
            ]


def input_mean_std(feats, lst, mean_index, std_index):
    if np.count_nonzero(~np.isnan(lst)) > 0:
        feats[mean_index] = np.nanmean(lst)
        feats[std_index] = np.nanstd(lst)
    


def extract1(comment):
    ''' This function extracts features from a single comment

    Parameters:
        comment : string, the body of a comment (after preprocessing)

    Returns:
        feats : numpy Array, a 173-length vector of floating point features (only the first 29 are expected to be filled, here)
    '''    
    feats = np.zeros(173)
    s = comment
    # TODO: Extract features that rely on capitalization.
    # Number of tokens in uppercase (≥ 3 letters long)
    feats[0] = len(re.compile(r"([A-Z]{3,})/[A-Z]{2,4}").findall(s))

    # TODO: Lowercase the text in comment. Be careful not to lowercase the tags. (e.g. "Dog/NN" -> "dog/NN").

    for word in re.findall("[A-Z]+/", s):
        s = s.replace(word, word.lower())

    # TODO: Extract features that do not rely on capitalization.

    regex = re.compile(r"(?<=\b)(?:" + r"|".join(FIRST_PERSON_PRONOUNS) + r")(?=\/|\b)")
    feats[1] = len(regex.findall(s))

    regex = re.compile(r"(?<=\b)(?:" + r"|".join(SECOND_PERSON_PRONOUNS) + r")(?=\/|\b)")
    feats[2] = len(regex.findall(s))

    regex = re.compile(r"(?<=\b)(?:" + r"|".join(THIRD_PERSON_PRONOUNS) + r")(?=\/|\b)")
    feats[3] = len(regex.findall(s))

    # Coordinating Conjunctions
    regex = re.compile(r"(?<=\b)(?:/CC)(?=\b)")
    feats[4] = len(regex.findall(s))

    # Number of past-tense verbs
    regex = re.compile(r"(?<=\b)(?:/VBD)(?=\b)")
    feats[5] = len(regex.findall(s))

    # Number of future tense. 
    regex = re.compile(r"(?:(?:going|gonna|will|'ll))")
    p1 = len(regex.findall(s) )
    # 'll, will, gonna, going+to+VB'
    regex = re.compile(r"(?:go/VBG\s+to/[A-Z]{2,}\s+\w*/VB|going\s+to/[A-Z]{2,}\s+\w*/VB)")
    p2 = len(regex.findall(s))
    feats[6] = p1 + p2

    # number of commas
    regex = re.compile(r",/,|(?<!/),")
    feats[7] = len(regex.findall(s))

    # number of multicharacter punctuations.
    regex = re.compile(r"(\.\.\.|[\$\#\?\!\:\;\.\(\)\"\',]{2,})")
    feats[8] = len(regex.findall(s))

    # n common nouns
    regex = re.compile(r"(?:/NN|/NNS)(?=\b)")
    feats[9] = len(regex.findall(s))

    # n proper nouns
    regex = re.compile(r"(?:/NNP|/NNPS)(?=\b)")
    feats[10] = len(regex.findall(s))

    # n adverbs
    regex = re.compile(r"(?:/RBR|/RBS|/RB)(?=\b)")
    feats[11] = len(regex.findall(s))

    # n wh-words
    regex = re.compile(r"(?:/WP\$\b|/WDT\b|/WRB\b|/WP\b)")
    feats[12] = len(regex.findall(s))

    # n slang words
    regex = re.compile(r"(?:\s|^)("+r"|".join(SLANG)+")(?:/[A-Z]{0,4})")
    feats[13] = len(regex.findall(s))

    # Average length in tokens.
    s = comment
    for word in re.findall("[A-Z]+/", s):
        s = s.replace(word, word.lower())
    words = s.split()
    sentences = comment.split("\n")[:-1]
    num_of_tokens = 0
    retrieved_words =[] 
    if len(words) and len(sentences) > 0:
        for sentence in sentences:
            num_of_tokens += len(sentence.split()) 
        feats[14] = num_of_tokens / len(sentences)

        for word in words:
            retrieved_words += re.findall(r"(/?\w+)(?=/)", word)

        # Average length of tokens, excluding punctuation-only tokens, in characters
        puncts = r"([\#\$\!\?\.\:\;\(\)\"\',\[\]/]{1,}|\.\.\.)/"
        non_punct_words = [word for word in words if re.match(puncts, word) is None]
        if len(non_punct_words) > 0:
            feats[15] = sum([len(word[:word.rfind('/')]) for word in non_punct_words]) \
                    / len(non_punct_words) 

        # Number of sentences
        feats[16] = len(sentences)

    AoA = []
    IMG = []
    FAM = []
    VMS = []
    AMS = []
    DMS = []
    if len(retrieved_words) > 0:
        for word in retrieved_words:
            if word in bgl_data:
                AoA.append(bgl_data[word][0])
                IMG.append(bgl_data[word][1])
                FAM.append(bgl_data[word][2])

        input_mean_std(feats, AoA, 17, 20)
        input_mean_std(feats, IMG, 18, 21)
        input_mean_std(feats, FAM, 19, 22)

    # Now onto Warringer norms.
        for word in retrieved_words:
            if word in warr_data:
                VMS.append(warr_data[word][0])
                AMS.append(warr_data[word][1])
                DMS.append(warr_data[word][2])

        input_mean_std(feats, VMS, 23, 26)
        input_mean_std(feats, AMS, 24, 27)
        input_mean_std(feats, DMS, 25, 28)

    return feats

    
def extract2(feat, comment_class, comment_id):
    ''' This function adds features 30-173 for a single comment.

    Parameters:
        feat: np.array of length 173
        comment_class: str in {"Alt", "Center", "Left", "Right"}
        comment_id: int indicating the id of a comment

    Returns:
        feat : numpy Array, a 173-length vector of floating point features (this 
        function adds feature 30-173). This should be a modified version of 
        the parameter feats.
    '''    
    # print('TODO')
    feat[29:] = file_data[comment_class][comment_id]
    return feat


    


def main(args):
    #Declare necessary global variables here. 

    #Load data
    data = json.load(open(args.input))
    feats = np.zeros((len(data), 173+1))

    # TODO: Call extract1 for each datatpoint to find the first 29 features. 
    # Add these to feats.
    # TODO: Call extract2 for each feature vector to copy LIWC features (features 30-173)
    # into feats. (Note that these rely on each data point's class,
    # which is why we can't add them in extract1).
    # print('TODO')

    for i, comment in enumerate(data):
        feats[i, :173] = extract1(comment['body'])
        feats[i, :173] = extract2(feats[i, :173], comment['cat'], comment['id'])
        feats[i, -1] = class_data[comment['cat']] 


    np.savez_compressed(args.output, feats)

    
if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("-i", "--input", help="The input JSON file, preprocessed as in Task 1", required=True)
    parser.add_argument("-p", "--a1_dir", help="Path to csc401 A1 directory. By default it is set to the cdf directory for the assignment.", default="/u/cs401/A1/")
    args = parser.parse_args()        

    main(args)


import os
import sys
import numpy as np
import re



dataDir = '/u/cs401/A3/data/'

def Levenshtein(r, h):
    """                                                                         
    Calculation of WER with Levenshtein distance.                               
                                                                                
    Works only for iterables up to 254 elements (uint8).                        
    O(nm) time ans space complexity.                                            
                                                                                
    Parameters                                                                  
    ----------                                                                  
    r : list of strings                                                                    
    h : list of strings                                                                   
                                                                                
    Returns                                                                     
    -------                                                                     
    (WER, nS, nI, nD): (float, int, int, int) WER, number of substitutions, insertions, and deletions respectively
                                                                                
    Examples                                                                    
    --------                                                                    
    >>> wer("who is there".split(), "is there".split())                         
    0.333 0 0 1                                                                           
    >>> wer("who is there".split(), "".split())                                 
    1.0 0 0 3                                                                           
    >>> wer("".split(), "who is there".split())                                 
    Inf 0 3 0                                                                           
    """
    r = ['<s>'] + r + ['</s>']
    h = ['<s>'] + h + ['</s>']
    num_rows = len(r)
    num_cols = len(h)
    A = np.zeros((num_rows, num_cols))

    # Initialization
    A[0] = np.arange(num_cols)
    A[:,0] = np.arange(num_rows)

    # Induction
    for i in range(1, num_rows):
        for j in range(1, num_cols):
            delete = A[i-1, j]
            insert = A[i, j-1]
            replace = A[i-1, j-1]
            if r[i] == h[j]:
                A[i,j] = A[i-1, j-1]
            else:
                A[i,j] = min(delete, insert, replace) + 1

    # Finding the minimum path and count.
    i = A.shape[0] - 1
    j = A.shape[1] - 1
    counts = {'sub': 0, 'ins': 0, 'del': 0}
    while i > 0 or j > 0:
        # index 0 - substitution, 1 - insertion, 2 - deletion.
        if i > 0 and j > 0:
            values = [A[i-1,j-1], A[i,j-1], A[i-1,j]]
        elif i <= 0 and j > 0:
            values = [np.inf, A[i, j-1], np.inf]
        elif i > 0 and j <= 0:
            values = [np.inf, np.inf, A[i-1, j]]
        else:
            break

        ind = values.index(min(values))
        if ind == 0:
            if A[i-1, j-1] == A[i,j] - 1:
                counts['sub'] += 1
            i -= 1
            j -= 1
        elif ind == 1:
            counts['ins'] += 1
            j -=1
        else:
            counts['del'] += 1
            i -= 1

    return [A[-1,-1]/(A.shape[0] - 2), counts['sub'], counts['ins'], counts['del']]

def preproc(line):
    line = re.sub("<[A-Z]+>", "", line)
    line = line.lower() # Make in lower case.
    return re.sub(r"[^\w\[\] ]+", "", line).split()[2:]


def format_output(output_info):
    s = "{} {} {} {} S:{}, I:{}, D:{}"
    s = s.format(*output_info)
    print(s)

if __name__ == "__main__":
    # print( 'TODO' ) 
    # sys.stdout = open('asrDiscussion.txt', 'w')
    goog = []
    # goog_s = []
    # goog_i = []
    # goog_d = []
    kald = []
    # kald_s = []
    # kald_i = []
    # kald_d = []
    for root, dirs, files in os.walk(dataDir):
        for speaker in dirs:
            # print(speaker)
            transcripts = os.path.join(root, speaker, 'transcripts.txt') 
            transcripts = open(transcripts, 'r').readlines()
            g_transcripts = os.path.join(root, speaker, 'transcripts.Google.txt') 
            g_transcripts = open(g_transcripts, 'r').readlines()
            k_transcripts = os.path.join(root, speaker, 'transcripts.Kaldi.txt')
            k_transcripts = open(k_transcripts, 'r').readlines()
            
            if len(transcripts) == 0:
                print("{} - reference trasncripts empty".format(speaker))
                print("")
                continue

            # calculate the values for google first
            for i, r in enumerate(transcripts):
                # For debug purpose
                g_output_info = []
                k_output_info = []
                r = preproc(r)
                g_output_info += Levenshtein(r, preproc(g_transcripts[i]))
                goog.append(g_output_info[0])
                # goog_s.append(g_output_info[1])
                # goog_i.append(g_output_info[2])
                # goog_d.append(g_output_info[3])
                g_output_info = [speaker, 'Google', i] + g_output_info
                format_output(g_output_info)

                k_output_info += Levenshtein(r, preproc(k_transcripts[i]))
                kald.append(k_output_info[0])
                # kald_s.append(k_output_info[1])
                # kald_i.append(k_output_info[2])
                # kald_d.append(k_output_info[3])
                k_output_info = [speaker, 'Kardi', i] + k_output_info 
                format_output(k_output_info)
                print("")


    goog = np.array(goog)
    kald = np.array(kald)
    print("Google has a mean of: {} and standard deviation of {}.".format(np.mean(goog), np.var(goog)))
    # print("\tsubstitution mean: {}".format(np.mean(goog_s)))
    # print("\tinsertion mean: {}".format(np.mean(goog_i)))
    # print("\tdeletion mean: {}".format(np.mean(goog_d)))

    print("Kaldi has a mean of: {} and standard deviation of {}.".format(np.mean(kald), np.var(kald)))
    # print("\tsubstitution mean: {}".format(np.mean(kald_s)))
    # print("\tinsertion mean: {}".format(np.mean(kald_i)))
    # print("\tdeletion mean: {}".format(np.mean(kald_d)))

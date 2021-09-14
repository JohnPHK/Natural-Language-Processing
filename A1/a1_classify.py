#  * This code is provided solely for the personal and private use of students
#  * taking the CSC401 course at the University of Toronto. Copying for purposes
#  * other than this use is expressly prohibited. All forms of distribution of
#  * this code, including but not limited to public repositories on GitHub,
#  * GitLab, Bitbucket, or any other online platform, whether as given or with
#  * any changes, are expressly prohibited.
#  
#  * All of the files in this directory and all subdirectories are:
#  * Copyright (c) 2020 Frank Rudzicz

import argparse
import os
from scipy.stats import ttest_rel
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold 
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier  
from sklearn.utils import shuffle

# set the random state for reproducibility 
import numpy as np
np.random.seed(401)

classifiers = [
        SGDClassifier(), 
        GaussianNB(), 
        RandomForestClassifier(n_estimators=10, max_depth=5), 
        MLPClassifier(alpha=0.02), 
        AdaBoostClassifier()
        ]

cls_dct = {
        0: "SGDClassifier",
        1: "GaussianNB",
        2: "RandomForestClassifier",
        3: "MLPClassifier",
        4: "AdaBoostClassifier"
        }

def accuracy(C):
    ''' Compute accuracy given Numpy array confusion matrix C. Returns a floating point value '''
    denom = np.sum(C)
    return np.trace(C) / denom if denom != 0 else 0


def recall(C):
    ''' Compute recall given Numpy array confusion matrix C. Returns a list of floating point values '''
    # print ('TODO')
    denom = np.sum(C, axis=1) #Summing the elements within each rows.
    return np.divide(np.diagonal(C), denom, out=np.zeros(4), where=denom!=0)



def precision(C):
    ''' Compute precision given Numpy array confusion matrix C. Returns a list of floating point values '''
    denom = np.sum(C, axis=0)
    return np.divide(np.diagonal(C), denom, out=np.zeros(4), where=denom!=0)
    

def class31(output_dir, X_train, X_test, y_train, y_test):
    ''' This function performs experiment 3.1
    
    Parameters
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes

    Returns:      
       i: int, the index of the supposed best classifier
    '''
    print('TODO Section 3.1')
    highest_acc = 0
    iBest = 0
    
    with open(f"{output_dir}/a1_3.1.txt", "w") as outf:
        # For each classifier, compute results and write the following output:
        for i in range(len(classifiers)):
            classifiers[i].fit(X_train, y_train)
            conf_matrix = confusion_matrix(y_test, classifiers[i].predict(X_test))
            acc = accuracy(conf_matrix)
            if acc > highest_acc:
                highest_acc = acc
                iBest = i
            rec = recall(conf_matrix)
            prec = precision(conf_matrix)
            classifier_name = cls_dct[i] 

            outf.write(f'Results for {classifier_name}:\n')  # Classifier name
            outf.write(f'\tAccuracy: {acc:.4f}\n')
            outf.write(f'\tRecall: {[round(item, 4) for item in rec]}\n')
            outf.write(f'\tPrecision: {[round(item, 4) for item in prec]}\n')
            outf.write(f'\tConfusion Matrix: \n{conf_matrix}\n\n')

    return iBest


def class32(output_dir, X_train, X_test, y_train, y_test, iBest):
    ''' This function performs experiment 3.2
    
    Parameters:
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       iBest: int, the index of the supposed best classifier (from task 3.1)  

    Returns:
       X_1k: numPy array, just 1K rows of X_train
       y_1k: numPy array, just 1K rows of y_train
   '''
    print('TODO Section 3.2')
    
    increments = [1000, 5000, 10000, 15000, 20000]
    with open(f"{output_dir}/a1_3.2.txt", "w") as outf:
        # For each number of training examples, compute results and write
        # the following output:
        bst_cls = classifiers[iBest]
        for num_train in increments:
            X_inc = X_train[:num_train]
            y_inc = y_train[:num_train]
            bst_cls.fit(X_inc, y_inc)
            acc = accuracy(confusion_matrix(y_test, bst_cls.predict(X_test)))
            outf.write(f'{num_train}: {acc:.4f}\n')

    X_train, y_train = shuffle(X_train, y_train)
    X_1k = X_train[:1000]
    y_1k = y_train[:1000]

    return (X_1k, y_1k)


def class33(output_dir, X_train, X_test, y_train, y_test, i, X_1k, y_1k):
    ''' This function performs experiment 3.3
    
    Parameters:
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)  
       X_1k: numPy array, just 1K rows of X_train (from task 3.2)
       y_1k: numPy array, just 1K rows of y_train (from task 3.2)
    '''
    print('TODO Section 3.3')
    
    with open(f"{output_dir}/a1_3.3.txt", "w") as outf:
        # Prepare the variables with corresponding names, then uncomment
        # this, so it writes them to outf.
        # for each number of features k_feat, write the p-values for
        # that number of features:
        k_feat = 5
        selector_5 = SelectKBest(f_classif, k=k_feat)
        X_new = selector_5.fit_transform(X_train, y_train)
        feats_32k = selector_5.get_support(True) # Get features for 32k
        p_values = selector_5.pvalues_
        outf.write(f'{k_feat} p-values: {[round(pval, 4) for pval in p_values]}\n')

        k_feat = 50
        selector_50 = SelectKBest(f_classif, k=k_feat)
        selector_50.fit(X_train, y_train)
        p_values = selector_50.pvalues_
        outf.write(f'{k_feat} p-values: {[round(pval, 4) for pval in p_values]}\n')

        # Start for 32_k save it up for later
        bst_cls = classifiers[i]
        bst_cls.fit(X_new, y_train)
        C = confusion_matrix(y_test, bst_cls.predict(selector_5.transform(X_test)))
        accuracy_full = accuracy(C)

        # Here deal with 1_k
        X_new = selector_5.fit_transform(X_1k, y_1k)
        feats_1k = selector_5.get_support(True) # Get features for 1k 
        bst_cls.fit(X_new, y_1k)
        C = confusion_matrix(y_test, bst_cls.predict(selector_5.transform(X_test)))
        accuracy_1k = accuracy(C)
        outf.write(f'Accuracy for 1k: {accuracy_1k:.4f}\n')
        outf.write(f'Accuracy for full dataset: {accuracy_full:.4f}\n')

        feature_intersection = np.intersect1d(feats_1k, feats_32k, return_indices=True)
        outf.write(f'Chosen feature intersection: {feature_intersection}\n')

        top_5 = feats_32k
        outf.write(f'Top-5 at higher: {top_5}\n')


def class34(output_dir, X_train, X_test, y_train, y_test, i):
    ''' This function performs experiment 3.4
    
    Parameters
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)  
        '''
    print('TODO Section 3.4')
    kfold_accuracies_p = {}
    
    with open(f"{output_dir}/a1_3.4.txt", "w") as outf:
        # Prepare kfold_accuracies, then uncomment this, so it writes them to outf.
        # for each fold:
        X = np.concatenate((X_train, X_test), axis=0)
        Y = np.concatenate((y_train, y_test), axis=0)
        kf = KFold(shuffle=True)
        for cls_index in range(len(classifiers)):
            kfold_accuracies = []
            classifier = classifiers[cls_index]
            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index], X[test_index] 
                y_train, y_test = Y[train_index], Y[test_index] 
                classifier.fit(X_train, y_train)
                C = confusion_matrix(y_test, classifier.predict(X_test))
                acc = accuracy(C)
                kfold_accuracies += [acc]
            outf.write(f'Kfold Accuracies: {[round(acc, 4) for acc in kfold_accuracies]}\n')
            kfold_accuracies_p[cls_index] = kfold_accuracies
        p_values = []
        for cls_index in range(len(classifiers)):
            if cls_index != i:
                S , pvalue= ttest_rel(kfold_accuracies_p[cls_index], kfold_accuracies_p[i])
                p_values += [pvalue]
            else:
                pass
        outf.write(f'p-values: {[round(pval, 4) for pval in p_values]}\n')


    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="the input npz file from Task 2", required=True)
    parser.add_argument(
        "-o", "--output_dir",
        help="The directory to write a1_3.X.txt files to.",
        default=os.path.dirname(os.path.dirname(__file__)))
    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # TODO: load data and split into train and test.
    npz_file = np.load(args.input)
    data = npz_file[npz_file.files[0]]

    # Split the data 2:8
    features = data[:, :-1]
    cat = data[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(features, 
                                                        cat, 
                                                        test_size=0.2)
    # TODO : complete each classification experiment, in sequence.
    iBest = class31(args.output_dir, X_train, X_test, y_train, y_test)
    X_1k, y_1k = class32(args.output_dir, X_train, X_test, y_train, y_test, iBest)
    class33(args.output_dir, X_train, X_test, y_train, y_test, iBest, X_1k, y_1k)
    class34(args.output_dir, X_train, X_test, y_train, y_test, iBest)



#! /usr/bin/python
#
# Implemented by Xuyun Zhang (email: xuyun.zhang@auckland.ac.nz). Copyright reserved.
#

import numpy as np
import pandas as pd
import csv
import time
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import IsolationForest
from sklearn.metrics.pairwise import cosine_similarity
from detectors import VSSampling
from detectors import Bagging
from detectors import LSHForest
from detectors import E2LSH, KernelLSH, AngleLSH


rng = np.random.RandomState(42)
num_ensemblers = 10

# data = pd.read_csv('dat/glass.csv', header=None)
data = pd.read_csv('dat/glass.csv', header=None) #data is a dataframe object
X = data.as_matrix()[:, :-1].tolist()
ground_truth = data.as_matrix()[:, -1].tolist()
#("sklearn.ISO", IsolationForest(random_state=rng)),
classifiers = [("ALSH", LSHForest(num_ensemblers, VSSampling(num_ensemblers), AngleLSH())),
			   ("L1SH", LSHForest(num_ensemblers, VSSampling(num_ensemblers), E2LSH(norm=1))),
			   ("L2SH", LSHForest(num_ensemblers, VSSampling(num_ensemblers), E2LSH(norm=2))),
			   ("KLSH", LSHForest(num_ensemblers, VSSampling(num_ensemblers), KernelLSH()))]

# classifiers = [("L1SH", LSHForest(num_ensemblers, VSSampling(num_ensemblers), E2LSH(norm=1)))

Deep_Layers = []
toStop = False
layer_index = 0
prev_y_pred = []
while toStop != True and  (layer_index) < 25:
    print "layer " + str(layer_index)
    cur_lay_clfs = []
    y_pred_list = []
    for i, (clf_name, clf) in enumerate(classifiers):
        # print "	"+clf_name+":"
        # prediction stage
        # start_time = time.time()
        clf.fit(X)
        cur_lay_clfs.append(clf)
        # train_time = time.time()- start_time
        # evaluation stage
        y_pred = clf.decision_function(X).ravel()
        y_pred_list.append(y_pred)
        # test_time = time.time()- start_time - train_time

    # Find the average anomaly score
    avg_anomaly_scores = []
    for i in range(len(X)):
        sum_score = 0.0
        for j in range(len(y_pred_list)):
            sum_score = sum_score + y_pred_list[j][i]
        avg_anomaly_scores.append(sum_score/len(y_pred_list))

    # Check to see if the avg anomaly score changes, using cosine similarity
    if layer_index != 0:
        # similarity_score = cosine_similarity(np.asarray(avg_anomaly_scores).reshape(-1,1), np.asarray(prev_y_pred).reshape(-1,1))
        similarity_score = cosine_similarity([avg_anomaly_scores], [prev_y_pred])
        print similarity_score
        if False:
            toStop = True
        else:
            # Remove the first element in each row
            for indx in range(len(X)):
                X[indx][0] = avg_anomaly_scores[indx]
                # Append the avg anomaly score to the dataset to train in the next layer
            # X = np.c_[avg_anomaly_scores.tolist(), X]
    else:
        X = np.c_[avg_anomaly_scores, X]
    # store the current layer of classifiers
    Deep_Layers.append(cur_lay_clfs)
    layer_index = layer_index + 1
    prev_y_pred =avg_anomaly_scores

# auc = roc_auc_score(ground_truth, -1.0*y_pred)
auc = roc_auc_score(ground_truth, y_pred)
print "AUC:	", auc
# print "Training time:	", train_time
# print "Testing time:	", test_time
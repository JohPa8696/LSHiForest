#! /usr/bin/python
#
# Implemented by Xuyun Zhang (email: xuyun.zhang@auckland.ac.nz). Copyright reserved.
#

import numpy as np
import pandas as pd
import csv
import time
from sklearn.metrics import roc_auc_score
import time
import matplotlib.pyplot as mpl
from sklearn.ensemble import IsolationForest

from detectors import VSSampling
from detectors import Bagging
from detectors import LSHForest
from detectors import E2LSH, KernelLSH, AngleLSH


rng = np.random.RandomState(42)
num_ensemblers = 10
num_layers = 2

# data = pd.read_csv('dat/glass.csv', header=None)
data = pd.read_csv('dat/glass.csv', header=None) #data is a dataframe object
X = data.as_matrix()[:, :-1].tolist()
ground_truth = data.as_matrix()[:, -1].tolist()
#
# classifiers = [("sklearn.ISO", IsolationForest(random_state=rng)),
# 			   ("ALSH", LSHForest(num_ensemblers, VSSampling(num_ensemblers), AngleLSH())),
# 			   ("L1SH", LSHForest(num_ensemblers, VSSampling(num_ensemblers), E2LSH(norm=1))),
# 			   ("L2SH", LSHForest(num_ensemblers, VSSampling(num_ensemblers), E2LSH(norm=2))),
# 			   ("KLSH", LSHForest(num_ensemblers, VSSampling(num_ensemblers), KernelLSH()))]

classifiers = [ ("L2SH", LSHForest(num_ensemblers, VSSampling(num_ensemblers), E2LSH(norm=1)))]
results =[]
# for ind in range(100):
temp = []
temp = X[:]
for i, (clf_name, clf) in enumerate(classifiers):
	print "	"+clf_name+":"
	# prediction stage
	start_time = time.time()
	prev_predictions = []
	for layer_index in range(num_layers):
		print "layer " + str(layer_index)
		if layer_index != 0:
			clf.fit(temp, prev_predictions)
		else:
			clf.fit(temp)
		clf.get_avg_branch_factor()
		# print "LSH Forest:"
		# clf.display()  # CONSOLE OUTPUT
		# print "-------------"
		# evaluation stage
		y_pred = clf.decision_function(temp).ravel()
		if( i < num_layers - 1 ):
			y_pred = y_pred.tolist()
		temp = np.c_[y_pred, temp].tolist()
		prev_predictions = y_pred
	train_time = time.time()- start_time
	test_time = time.time()- start_time - train_time
	auc = roc_auc_score(ground_truth, y_pred)
	results.append(auc)
	print "AUC:	", auc
	mpl.plot(range(len(y_pred)), y_pred, 'ro')
	# print "Training time:	", train_time
	# print "Testing time:	", test_time

with open("deep1.csv","r+") as filerw:
	resultWriter = csv.writer(filerw)
	resultWriter.writerow(results)

filerw.close()
# mpl.boxplot(results)
# mpl.show()
print results
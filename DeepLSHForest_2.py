#! /usr/bin/python

import numpy as np
import pandas as pd
import csv
import time
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as mpl
from sklearn.ensemble import IsolationForest

from detectors import VSSampling
from detectors import Bagging
from detectors import LSHForest
from detectors import E2LSH, KernelLSH, AngleLSH


rng = np.random.RandomState(42)
num_ensemblers = 10
num_layers = 5

# data = pd.read_csv('dat/glass.csv', header=None)
data = pd.read_csv('dat/glass.csv', header=None) #data is a dataframe object
X = data.as_matrix()[:, :-1].tolist()
ground_truth = data.as_matrix()[:, -1].tolist()
#
classifiers = [("ALSH", LSHForest(num_ensemblers, VSSampling(num_ensemblers), AngleLSH()))]
results =[]
for ind in range(100):
	temp = []
	temp = X[:]
	for i, (clf_name, clf) in enumerate(classifiers):
		# print "	"+clf_name+":"
		# prediction stage
		start_time = time.time()
		for ind in range(num_layers):
			clf.fit(temp)
			# print "LSH Forest:"
			# clf.display()  # CONSOLE OUTPUT
			# print "-------------"
			# evaluation stage
			y_pred = clf.decision_function(temp).ravel()
			if(ind != 0):
				for row in temp:
					row.pop(0)
			if(ind < num_layers-1):
				y_pred = y_pred.tolist()
			temp = np.c_[y_pred,temp].tolist()
		train_time = time.time()- start_time
		test_time = time.time()- start_time - train_time
		auc = roc_auc_score(ground_truth, -1.0*y_pred)
		results.append(auc*100)
		# print "AUC:	", auc
		# print "Training time:	", train_time
		# print "Testing time:	", test_time
with open("deep2.csv","w") as filerw:
	resultWriter = csv.writer(filerw)
	resultWriter.writerow(results)

filerw.close()
mpl.boxplot(results)
mpl.show()
print results
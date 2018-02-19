#! /usr/bin/python
#
# Implemented by Xuyun Zhang (email: xuyun.zhang@auckland.ac.nz). Copyright reserved.
#

import numpy as np
import pandas as pd
import csv
import time
import matplotlib.pyplot as mpl
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import IsolationForest

from detectors import VSSampling
from detectors import Bagging
from detectors import LSHForest
from detectors import E2LSH, KernelLSH, AngleLSH
from detectors import SciForest

rng = np.random.RandomState(42)
num_ensemblers = 100

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

classifiers = [("L1SH", LSHForest(num_ensemblers, VSSampling(num_ensemblers), E2LSH(norm=1)))]
results = []

for i, (clf_name, clf) in enumerate(classifiers):
	print "	"+clf_name+":"
	# prediction stage
	start_time = time.time()
	clf.fit(X)
#	print clf.display()
	clf.get_avg_branch_factor()
	train_time = time.time() - start_time
	# evaluation stage
	y_pred = clf.decision_function(X).ravel()
	# y_pred.sort()
	# print y_pred
	test_time = time.time()- start_time - train_time
	auc = roc_auc_score(ground_truth, -1.0*y_pred)
	results.append(auc*100)
	print "AUC:	", auc
	mpl.plot(range(len(y_pred)), y_pred,'ro')
	# print "Training time:	", train_time
	# print "Testing time:	", test_time

# with open("demo.csv","w") as filerw:
# 	resultWriter = csv.writer(filerw)
# 	resultWriter.writerow(results)
#
# filerw.close();
mpl.boxplot(results)
mpl.show()
print results
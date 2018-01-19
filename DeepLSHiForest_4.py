#! /usr/bin/python

import numpy as np
import pandas as pd
import csv
import threading
import time
import multiprocessing
from multiprocessing import Pool
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import IsolationForest
from sklearn.metrics.pairwise import cosine_similarity
from numpy import linalg as LA
from detectors import VSSampling
from detectors import Bagging
from detectors import LSHForest
from detectors import E2LSH, KernelLSH, AngleLSH


def train_classifier(arg):
	classifier, data= arg
	clf_name, clf = classifier
	print "	" + clf_name + ":"
	# Build the clf/construct LSH forest
	clf.fit(data)
	return clf

def get_prediction(arg):
	clf, data = arg
	# evaluation stage
	y_pred = clf.decision_function(data).ravel()
	# Copy prediction result to another array
	sorted = list(y_pred)
	# Sort the array
	sorted.sort(reverse=True)
	# Function thresholding
	y_pred_var = np.var(y_pred)
	y_pred_mean = np.mean(y_pred)
	print "	mean: " + str(y_pred_mean) + " - variance: " + str(y_pred_var)
	# Calculate the number of instances in the 5 or 95 percentile
	# Get the instances with the lowest y_pred
	# percentile5 = sorted[0:int(len(y_pred) * 0.08 - 1.0)]
	num_outliers =0;
	bias_pred = []
	for yi in y_pred:
		outlier_score = yi - y_pred_mean - y_pred_var * int(len(y_pred) * 0.08 - 1.0)
		if outlier_score >= 0:
			bias_pred.append(1)
			num_outliers += 1
		else:
			bias_pred.append(0)
	print "	num_outliers: " +str(num_outliers)
	# Assign 1 to the instances in the 5 percentile and -1 to the rest
	# THIS IS A VERY NAIVE WAY OF INTRODUCING BIAS
	# LOOK FOR BETTER WAYS
	# for y_pred_indx in range(len(y_pred)):
	# 	if y_pred[y_pred_indx] in percentile5:
	# 		bias_pred.append(1)
	# 	else:
	# 		bias_pred.append(0)
	return (bias_pred, y_pred.tolist())


if __name__ == '__main__':

	rng = np.random.RandomState(42)
	# Number of trees
	num_ensemblers = 100

	# data = pd.read_csv('dat/glass.csv', header=None)
	data = pd.read_csv('dat/glass.csv', header=None) #data is a dataframe object
	X = data.as_matrix()[:, :-1].tolist()
	ground_truth = data.as_matrix()[:, -1].tolist()

	results =[]

	Deep_Layers = []
	toStop = False
	layer_index = 0
	prev_y_pred = []
	prev_similarity_score = 1
	while toStop != True and (layer_index) < 5:
		start_time = time.time()
		# Initialize classifiers
		print "layer " + str(layer_index)
		classifiers = [("KLSH1_lay_" + str(layer_index), LSHForest(num_ensemblers, VSSampling(num_ensemblers), KernelLSH(nbits=1, kernel='puk'))),
					   ("KLSH2_lay_" + str(layer_index), LSHForest(num_ensemblers, VSSampling(num_ensemblers), KernelLSH(nbits=2, kernel='puk'))),
					   ("KLSH3_lay_" + str(layer_index), LSHForest(num_ensemblers, VSSampling(num_ensemblers), KernelLSH(nbits=3, kernel='puk'))),
					   ("KLSH4_lay_" + str(layer_index), LSHForest(num_ensemblers, VSSampling(num_ensemblers), KernelLSH(nbits=4, kernel='puk'))),
					   ("KLSH5_lay_" + str(layer_index), LSHForest(num_ensemblers, VSSampling(num_ensemblers), KernelLSH(nbits=5, kernel='puk')))]
		# classifiers = [("ALSH", LSHForest(num_ensemblers, VSSampling(num_ensemblers), AngleLSH())),
		# 			   ("ALSH", LSHForest(num_ensemblers, VSSampling(num_ensemblers), AngleLSH())),
		# 			   ("ALSH", LSHForest(num_ensemblers, VSSampling(num_ensemblers), AngleLSH())),
		# 			   ("ALSH", LSHForest(num_ensemblers, VSSampling(num_ensemblers), AngleLSH())),
		# 			   ("ALSH", LSHForest(num_ensemblers, VSSampling(num_ensemblers), AngleLSH()))]

		print len(X[0])
		# Initialize layer variable containers
		cur_lay_clfs = []
		y_pred_scaled = []
		y_pred = []
		similarity_score = 1
		threads = []
		# Create a pool of processes for each of the classifiers
		# Then train the classifiers
		p = Pool(len(classifiers))
		print "	Training"
		cur_lay_clfs = p.map(train_classifier,((clf, X) for clf in classifiers))
		print "	Getting Prediction"
		predictions = p.map(get_prediction,((clf,X) for clf in cur_lay_clfs))
		p.close()

		for pred in predictions:
			y_pred_scaled.append(pred[0])
			y_pred.append(pred[1])
		print y_pred_scaled
		# Check to see if the avg anomaly score changes, using cosine similarity
		if layer_index != 0:
			# Calculate the Frobenius distance of 2 matrix, determine the similarity. i.e: "Fro norm"
			similarity_score = LA.norm(np.asarray(y_pred_scaled) - np.asarray(prev_y_pred),'fro')
			print "Layer " +str(layer_index) + " - Similarity score:" +str( similarity_score)
			# if (similarity_score/prev_similarity_score > 0.95):
			if False:
				toStop = True
			else:
				# Concatenate the prediction to X
				transformed_y_pred_mat = map(list, zip(*y_pred_scaled))
				# X = np.c_[transformed_y_pred_mat, X]

				# Remove the first element in each row
				for i in range(len(X)):
					for j in range(len(classifiers)):
						X[i][j] = transformed_y_pred_mat[i][j]
		else:
			transformed_y_pred_mat = map(list, zip(*y_pred_scaled))
			X = np.c_[transformed_y_pred_mat, X]
			# X = np.c_[avg_anomaly_scores, X]
		# store the current layer of classifiers
		run_time = time.time() - start_time
		print " Execution time: " + str(run_time)
		Deep_Layers.append(cur_lay_clfs)
		layer_index = layer_index + 1
		prev_y_pred = list(y_pred_scaled)
		prev_similarity_score = similarity_score

	# auc = roc_auc_score(ground_truth, -1.0*y_pred)
	for final_pred in y_pred:
		auc = roc_auc_score(ground_truth, final_pred)
		results.append(auc*100)
		print "AUC:	", auc
	# Get the average prediction
	avg_score =[]
	for i in range(len(X)):
		sum_score = 0
		for j in range(len(classifiers)):
			sum_score = sum_score + y_pred[j][i]
		avg_score.append(sum_score/len(classifiers))
	auc = roc_auc_score(ground_truth, avg_score)
	print "Average AUC:	", auc
	# print "Training time:	", train_time
	# print "Testing time:	", test_time
	# with open("deep3.csv", "w") as filerw:
	#     resultWriter = csv.writer(filerw)
	#     resultWriter.writerow(results)
	# filerw.close()


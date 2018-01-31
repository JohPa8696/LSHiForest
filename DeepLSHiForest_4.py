#! /usr/bin/python

import numpy as np
import pandas as pd
import csv
import threading
import time
import multiprocessing
from multiprocessing import Pool
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import Lasso,LassoCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics.pairwise import cosine_similarity
from numpy import linalg as LA
from detectors import VSSampling
from detectors import Bagging
from detectors import LSHForest
from detectors import E2LSH, KernelLSH, AngleLSH


class Result_Wrapper:
	def __init__(self,bias_pred, y_pred):
		self.bias_pred = bias_pred
		self.y_pred = y_pred
	def set_mse(self,mse):
		self.mse = mse

	def set_coef(self, coef):
		self.lasso_coef = coef


# def train_classifier(arg):
def train_classifier(classifier, data):
	# classifier, data= arg

	clf_name, clf = classifier
	print "	" + clf_name + ":"
	# Build the clf/construct LSH forest
	clf.fit(data)
	return clf

def get_prediction(clf, data):
	# clf, data = arg
	# evaluation stage
	y_pred = clf.decision_function(data).ravel()
	# Copy prediction result to another array
	sorted = list(y_pred)
	# Sort the array
	sorted.sort(reverse= True)
	# Function thresholding
	y_pred_var = np.var(y_pred)
	y_pred_mean = np.mean(y_pred)
	print "	mean: " + str(y_pred_mean) + " - variance: " + str(y_pred_var)
	# Calculate the number of instances in the 5 or 95 percentile
	# Get the instances with the lowest y_pred
	percentile5 = sorted[0:int(len(y_pred) * 0.1 - 1.0)]
	outliers =[]
	outliers_scores =[]
	num_outliers =0;
	bias_pred = []
	# for yi in y_pred:
	# 	outlier_score = yi - y_pred_mean - y_pred_var * int(len(y_pred) * 0.08 - 1.0)
	# 	if outlier_score >= 0:
	# 		bias_pred.append(1)
	# 		num_outliers += 1
	# 	else:
	# 		bias_pred.append(0)
	# print "	num_outliers: " +str(num_outliers)
	# Assign 1 to the instances in the 5 percentile and -1 to the rest
	# THIS IS A VERY NAIVE WAY OF INTRODUCING BIAS
	# LOOK FOR BETTER WAYS
	for y_pred_indx in range(len(y_pred)):
		if y_pred[y_pred_indx] in percentile5:
			outliers.append(data[y_pred_indx])
			outliers_scores.append(y_pred[y_pred_indx])
			bias_pred.append(1)
		else:
			bias_pred.append(0)

	# Do feature extraction
	result_wrapper = Result_Wrapper(bias_pred,y_pred.tolist())
	feature_selection(result_wrapper, data, bias_pred, outliers, outliers_scores)
	return result_wrapper

def feature_selection(result_wrapper, data, pseudo_target, outliers, outliers_scores):
	# outliers_scores = np.ones(len(outliers))
	# alphas = np.logspace(-2,0.5,10)
	# opt_alpha = alphas[0]
	# coef = []
	# min_mse = 1.0
	# for a in alphas:
	# 	X_train, X_test, y_train, y_test = train_test_split(data, pseudo_target, test_size=0.1)
	# 	lassoReg = Lasso(alpha=a, normalize=True)
	# 	lassoReg.fit(X_train, y_train)
	# 	pred = lassoReg.predict(X_test)
	# 	mse = np.mean((pred - y_test)**2)
	# 	print " mean square error: " + str(a) + " - "+str(mse)
	# 	print " lasso score: " + str(lassoReg.score(X_test,y_test))
	# 	if(mse < min_mse):
	# 		min_mse = mse
	# 		opt_alpha = a
	# 		coef =lassoReg.coef

	lassoCV = LassoCV(cv=10)
	### THIS IMPLEMENTATION USES ONLY THE INSTANCES THAT WAS REGARDED AS ABNOMALIES
	lassoCV.fit(outliers, outliers_scores)
	pred = lassoCV.predict(outliers)
	mse = np.mean((pred - outliers_scores) ** 2)
	### THIS IMPLEMENTATION APPENDS THE PREDEICTIONS TO THE DATA BEFORE LASSO REGRESSION
	# data  = np.c_[np.asarray(pseudo_target).reshape(-1,1),data]
	# lassoCV.fit(data,pseudo_target)
	# pred = lassoCV.predict(data)
	# mse = np.mean((pred - pseudo_target)**2)

	result_wrapper.set_mse(mse)
	result_wrapper.set_coef(lassoCV.coef_)
	print " mean square error: " +str(mse)
	print " alpha: " + str(lassoCV.alpha_)
	print " lasso score: " + str(lassoCV.score(data, pseudo_target))
	print lassoCV.coef_
	return lassoCV.coef_


if __name__ == '__main__':

	rng = np.random.RandomState(42)
	# Number of trees
	num_ensemblers = 100

	data = pd.read_csv('dat/isolet.csv', header=None)
	# data = pd.read_csv('dat/isolet.csv', header=None) #data is a dataframe object
	X = data.as_matrix()[:, :-1].tolist()
	ground_truth = data.as_matrix()[:, -1].tolist()

	results =[]

	Deep_Layers = []
	toStop = False
	layer_index = 0
	prev_y_pred = []
	prev_similarity_score = 1
	prev_mse = 1.0
	while toStop != True and (layer_index) < 20:
		start_time = time.time()
		# Initialize classifiers
		print "layer " + str(layer_index)
		# classifiers = [("KLSH1_lay_" + str(layer_index), LSHForest(num_ensemblers, VSSampling(num_ensemblers), KernelLSH(nbits=1, kernel='rbf'))),
		# 			   ("KLSH2_lay_" + str(layer_index), LSHForest(num_ensemblers, VSSampling(num_ensemblers), KernelLSH(nbits=2, kernel='rbf'))),
		# 			   ("KLSH3_lay_" + str(layer_index), LSHForest(num_ensemblers, VSSampling(num_ensemblers), KernelLSH(nbits=3, kernel='rbf'))),
		# 			   ("KLSH4_lay_" + str(layer_index), LSHForest(num_ensemblers, VSSampling(num_ensemblers), KernelLSH(nbits=4, kernel='rbf'))),
		# 			   ("KLSH5_lay_" + str(layer_index), LSHForest(num_ensemblers, VSSampling(num_ensemblers), KernelLSH(nbits=5, kernel='rbf')))]
		classifiers = [("KLSH1_lay_" + str(layer_index),
						LSHForest(num_ensemblers, VSSampling(num_ensemblers), KernelLSH(nbits=5, kernel='rbf')))]
		print len(X[0])
		# Initialize layer variable containers
		cur_lay_clfs = []
		y_pred_scaled = [] # values 1 and 0
		y_pred = []   # real prediction score
		lasso_coefs = []
		# similarity_score = 1
		# Create a pool of processes for each of the classifiers
		# Then train the classifiers
		p = Pool(len(classifiers))
		print "	Training"
		# cur_lay_clfs = p.map(train_classifier,((clf, X) for clf in classifiers))
		# Layer with only 1 clf
		cur_lay_clfs.append(train_classifier(classifiers[0],X))
		print "	Getting Prediction"
		# results = p.map(get_prediction,((clf,X) for clf in cur_lay_clfs))
		results.append(get_prediction(cur_lay_clfs[0],X))
		p.close()
		cur_mse = 0.0
		for res in results:
			y_pred_scaled.append(res.bias_pred)
			y_pred.append(res.y_pred)
			lasso_coefs.append(res.lasso_coef)
			cur_mse += res.mse

		print y_pred_scaled

		# Determine the feature to remove
		coef_scores = np.zeros(len(lasso_coefs[0]))
		for a in lasso_coefs:
			for i in range(len(a)):
				if (a[i] == 0.0):
					coef_scores[i] -= 1
				else:
					coef_scores[i] += 1
		remove_indx = []
		for i in range(len(coef_scores)):
			if coef_scores[i] <= 0:
				remove_indx.append(i)
		if len(remove_indx) > 0:
			X = np.delete(X, remove_indx, axis=1)

		# Check to see if the avg anomaly score changes, using cosine similarity
		if layer_index != 0:
			# Calculate the Frobenius distance of 2 matrix, determine the similarity. i.e: "Fro norm"
			# similarity_score = LA.norm(np.asarray(y_pred_scaled) - np.asarray(prev_y_pred),'fro')
			print "Layer " +str(layer_index) + " - Mean square Error:" +str(cur_mse/len(classifiers))
			# if (similarity_score/prev_similarity_score > 0.95):
			if cur_mse/len(classifiers) > prev_mse:
			# if False:
				toStop = True
			# else:
			# 	# Concatenate the prediction to X
			# 	transformed_y_pred_mat = map(list, zip(*y_pred_scaled))
			# 	# X = np.c_[transformed_y_pred_mat, X]
            #
			# 	# Replace the class label from the previous run
			# 	for i in range(len(X)):
			# 		for j in range(len(classifiers)):
			# 			X[i][j] = transformed_y_pred_mat[i][j]
		# else:
		transformed_y_pred_mat = map(list, zip(*y_pred_scaled))
		X = np.c_[transformed_y_pred_mat, X]
		# store the current layer of classifiers
		run_time = time.time() - start_time
		print " Execution time: " + str(run_time)
		Deep_Layers.append(cur_lay_clfs)
		layer_index = layer_index + 1
		prev_y_pred = list(y_pred_scaled)
		prev_mse = cur_mse/len(classifiers)

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


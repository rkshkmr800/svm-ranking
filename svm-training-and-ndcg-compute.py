##########################################
# Group Details                          #
# 17CS60R71 : Chanderki Rakesh kumar     #
# 17CS60R58 : Akhilesh kumar yadav       #
##########################################

# - * - coding: utf-8 - * -

import os, os.path
import codecs
import re
import math

from collections import defaultdict

from sklearn import preprocessing
from sklearn import svm
import numpy as np

### INPUT PARAMTERS	###

data_folder = "../data/"	#data folder
output_folder = "../output/"	#data folder
idfs_input_file = "idfs.txt" #present in data folder
training_relevance_file = "pa3.rel.train" #present in data folder
training_data_file = "pa3.signal.train" #present in data folder
new_training_data_file = "new_training_data_query_doc_pair" #present in data folder
doc_doc_data_file = "new_training_data_doc_doc_pair" #present in data folder
test_data_file = "test_data"	#test data to compute ndcg score
output_file = "output"	#output file of ndcg scores of queries in dev file

### end ###

classifier = svm.SVC(kernel='linear' , probability = True)	#defining the svm model

#comparator to sort doc-doc pairs
def comparator(x, y):	#for ranking two documents based on the class
    temp = np.matrix(x[0]) - np.matrix(y[0])
    temp = temp.tolist()[0]
    score = classifier.predict_proba(temp)	#predicting probabilty values
    score = score[0][0]	#probability score of class '-1'
    if score <= 0.5:	#if score < 0.5 => class '+1' else '-1'
    	return 1	#doc x comes before doc y
    else:	
    	return -1


###reading doc-doc pair training data ###
file_training_data = open(output_folder + doc_doc_data_file + ".txt", 'r')
training_data = file_training_data.read()

###output file###
output_file_ptr = open(output_folder + output_file + ".txt", 'w')
output_file_report = open(output_folder + "report.csv", 'w')
output_file_report.write("Query,NDCG Score\n")

records = training_data.split("\n")
records.pop(-1)

features_x = []	#feature vectors
feature_y = []	#scores of the vectors
for record in records:
	record = record.split(" | ")
	temp = []

	for item in record:
		if item[0]=='-':	#to handle negative float numbers
			item = item.split("-")
			try:
				temp.append((-1.0 * (float(item[1]))))
			except:
				temp.append(0)
		else:	#to handle normal values
			temp.append(float(item))
	
	score = int(temp[5])
	temp.pop(-1)
	features_x.append(temp)	#populate feature vectors
	feature_y.append(score)	#populate scores list

print "Training started"

classifier.fit(features_x,feature_y)	#training the data on svm model

print "Training finished"

###reading doc-doc pair training data ###
test_data_file = open(output_folder + test_data_file + ".txt", 'r')
test_data = test_data_file.read()

queries = test_data.split("###\n")
queries.pop(-1)

ndcg_sum = 0

#iterate over each query
for query in queries:
	records = query.split("\nquery : ")
	query = records[1]

	records = records[0].split("\n")

	rel_list = []
	doc_list = []

	#iterate over each vector of test data
	for record in records:
		record = record.split(" | ")
		temp = []
		doc_scores = []

		for item in record:
			if item[0]=='-':	#to handle negative float numbers
				item = item.split("-")
				try:
					temp.append((-1.0 * (float(item[1]))))
				except:
					temp.append(0)
			else:	#to handle normal values
				temp.append(float(item))

		relevance = temp[5]
		temp.pop(-1)

		rel_list.append(relevance)	#list of relevance scores of documents of the query
		
		doc_scores.append(temp)
		doc_scores.append(relevance)	

		doc_list.append(doc_scores)


	doc_list.sort(comparator)	#ranking in the order of probability of class of doc-doc pair
	rel_list.sort(reverse = True)	#optimal order of ranking

	#computing the dcg of our classifier
	iterator = 1
	sum = 0.0
	for item in doc_list:

		sum += (((2**item[1])-1)/math.log(1+iterator))
					
		iterator += 1

	dcg_actual = sum

	#computing the optimal dcg
	iterator = 1
	sum = 0.0
	for item in rel_list:
		sum += (((2**item)-1)/math.log(1+iterator))
		iterator += 1

	dcg_optimal = sum

	if dcg_optimal == 0:
		normalized_zk = 0
	else:
		normalized_zk = 1/dcg_optimal

	ndcg = dcg_actual*normalized_zk	#NDCG score

	ndcg_sum += ndcg

	output_file_ptr.write("query : "+query+" | NDCG score : "+ str(ndcg)+"\n")
	output_file_report.write(query+","+str(ndcg)+"\n")
	
	print "output file generated\ncheck the output folder for '"+output_file+".txt' "

	print "Average NDCG : ",ndcg_sum/120

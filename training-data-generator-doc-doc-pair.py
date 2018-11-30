##########################################
# Group Details                          #
# 17CS60R71 : Chanderki Rakesh kumar     #
# 17CS60R58 : Akhilesh kumar yadav       #
##########################################

import os, os.path
import codecs
import re

from collections import defaultdict

from sklearn import preprocessing
import numpy as np


### INPUT PARAMTERS	###

data_folder = "../data/"	#data folder
output_folder = "../output/"	#data folder
idfs_input_file = "idfs.txt" #present in data folder
training_relevance_file = "pa3.rel.train" #present in data folder
training_data_file = "pa3.signal.train" #present in data folder
new_training_data_file = "new_training_data_query_doc_pair" #present in data folder
doc_doc_data_file = "new_training_data_doc_doc_pair" #present in data folder

### end ###

doc_doc_pair_feature_vectors = []

###reading query-doc pair training data ###
file_training_data = open(output_folder + new_training_data_file + ".txt", 'r')
training_data = file_training_data.read()

new_training_data = open(output_folder + doc_doc_data_file + ".txt", 'w')

queries = training_data.split("###query###\n")
queries.pop(-1)	#empty last item
for query in queries:

	doc_query_vector_strings = query.split("\n")
	doc_query_vector_strings.pop(-1) #empty last item
	doc_query_vectors = []	#vectors of docs in query

	### converting features from string to float
	for doc_query_vector_string in doc_query_vector_strings:

		doc_query_vector = doc_query_vector_string.split(" | ")

		temp = []

		for item in doc_query_vector:
			if item[0]=='-':	#to handle negative float numbers
				item = item.split("-")
				try:
					temp.append((-1.0 * (float(item[1]))))
				except:
					temp.append(0)
			else:	#to handle normal values
				temp.append(float(item))

		doc_query_vectors.append(temp)	#insert to list of docs in query

	n = len(doc_query_vectors)

	#iterate over each doc and compute doc pair vector
	for i in range(0,n):
		for j in range((i+1),n):
			score = 0
			if doc_query_vectors[i][5] == doc_query_vectors[j][5]:	#if releavance matches skip
				continue

			if doc_query_vectors[i][5] > doc_query_vectors[j][5]:	#if relevance of doc1 > doc2 score = +1 else -1
				score = 1
			else:
				score = -1

			doc_doc_vector_1 = np.matrix(doc_query_vectors[i]) - np.matrix(doc_query_vectors[j])
			doc_doc_vector_1 = doc_doc_vector_1.tolist()[0]
			doc_doc_vector_1[5] = score

			doc_doc_vector_2 = np.matrix(doc_query_vectors[j]) - np.matrix(doc_query_vectors[i])
			doc_doc_vector_2 = doc_doc_vector_2.tolist()[0]
			doc_doc_vector_2[5] = score *(-1)

			doc_doc_pair_feature_vectors.append(doc_doc_vector_1)
			doc_doc_pair_feature_vectors.append(doc_doc_vector_2)

#writing the new traing data to file
for record in doc_doc_pair_feature_vectors:
	new_training_data.write(str(record[0])+" | "+str(record[1])+" | "+str(record[2])+" | "+str(record[3])+" | "+str(record[4])+" | "+str(record[5])+" \n")

print len(doc_doc_pair_feature_vectors),"records generated"
print "check",doc_doc_data_file+".txt file in output folder"
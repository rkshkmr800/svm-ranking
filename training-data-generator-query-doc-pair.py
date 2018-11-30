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

import nltk
from nltk import tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

### INPUT PARAMTERS	###

data_folder = "../data/"	#data folder
output_folder = "../output/"	#data folder
idfs_input_file = "idfs.txt" #present in data folder
training_relevance_file = "pa3.rel.train" #present in data folder
training_data_file = "pa3.signal.train" #present in data folder
new_training_data_file = "new_training_data_query_doc_pair" #present in data folder

### end ###

stop_words        =  list(stopwords.words('english'))	#to get stopwords from nltk corpus
punctuation_list  =  [".",";","{","}","[","]","(",")","!","@","#","-","_","--",",","%","/","'",":",'"',"\\","*","^","+",">","<","?","|","=","`","~"]	#to remove punctuations as tokens
stop_words.extend(punctuation_list)	#adding puntuitations to list

if not os.path.exists(data_folder):	#for data directory
	os.mkdir(data_folder)

if not os.path.exists(output_folder):	#for output directory
	os.mkdir(output_folder)


### creating idfs dictionary ###
file_idf = codecs.open(data_folder + idfs_input_file, 'r')
idfs = file_idf.readlines()

#to buld the idf values map
idf_values = {}
for idf in idfs:
	idf = idf.split(":")
	idf_values[idf[0]]=float(idf[1])


### creating relevance-scores dictionary ###
file_rel = codecs.open(data_folder + training_relevance_file, 'r')
rel_data = file_rel.read()
rel_data = rel_data.split("query: ")
rel_data.pop(0)

### to buld the relevance values map ###
rel_values = {}
for query in rel_data:

	query_data = query.replace("\n","")
	query_data = re.sub( '\s+', ' ', query_data ).split(" url: ")
	query = query_data[0]
	query_data.pop(0)

	url_dict = {}
	for url in query_data:
		url=url.split()
		url_dict[url[0]]=float(url[1]) 

	rel_values[query] = url_dict
### end of relevance scores map population ###


#to compute the term frequency vector for a string in general
def tf_vector(text):
	words = text.split()
	fq= defaultdict( int )
	for w in words:
		fq[w] += 1
	return fq

#to compute the tf-idf score for given query and term frequency vector of document
def tf_idf_score(query,document):

	query = set(query.split())
	document_tokens = set(document.keys())

	#list of common tokens
	tokens = list(document_tokens.intersection(query))

	idf_sum = 0

	for term in tokens:
		try:
			idf_sum += idf_values[term]*document[term]
		except:
			idf_sum += 0

	return idf_sum


###reading given training data ###
file_training_data = codecs.open(data_folder + training_data_file, 'r',  encoding='utf-8')
training_data = file_training_data.read()

### output training data file ###
new_training_data = open(output_folder + new_training_data_file + ".txt", 'w')
#new_training_data.write("url | title | header | body | anchor | relevance-score \n")

#new_training_data_csv = open(output_folder + new_training_data_file + ".csv" , 'w')
#new_training_data_csv.write("url , title , header , body , anchor , relevance-score \n")

#training data vector
train_data = []

#traing data relevance scores
relevance_data = []

#result count of each query
query_result_count = []

#getting list of queries and their results
queries = training_data.split("query: ")
queries.pop(0)
for query in queries:

	#getting list of results for particular query
	results = query.split("url: ")
	query = results[0]	#query string
	results.pop(0)

	query_result_count.append(len(results))

	#iterating over each results
	for result in results:
		result = result.replace("\n", ";")
		result = re.sub( '\s+', ' ', result ).strip()
		result = result.replace("; stanford_anchor_count:","|")

		result_data = result.split("; ")

		#url string
		url = result_data[0]
		url_string = url
		result_data.pop(0)

		#creating 2D array of components of result
		for i in range(0,len(result_data)):
			result_data[i]=result_data[i].split(":")

		header = ""
		body = ""
		weighted_sum = 0
		sum_of_weights = 0

		for result in result_data:

			if str(result[0])=="title":
				title = result[1]	#store title string

			elif str(result[0])=="header":
				header += (result[1]+" ")	#header string

			elif str(result[0])=="body_hits":
				tokens = result[1].split()
				word = tokens[0]
				tokens.pop()
				for token in tokens: 
					body += (word+" ")	#body string

			elif str(result[0])=="anchor_text":
				tokens = result[1].split("| ")
				tokens[1] = tokens[1].replace(";","")
				weight = int(tokens[1])
				#computing weighted score for multiple anchor texts where weight is frequency of anchor tags
				weighted_sum += (weight*tf_idf_score(query,tf_vector(tokens[0])))
				sum_of_weights += weight
	

		#parsing url string
		for token in punctuation_list:
			url = url.replace(token," ")

		url = re.sub( '\s+', ' ', url ).strip()

		url_vector = tf_vector(url)
		url_score = tf_idf_score(query,url_vector)
		### end url parse ###

		#parsing title string
		title_vector = tf_vector(title)
		title_score = tf_idf_score(query,title_vector)
		### end header parse ###

		#parsing header string
		header_vector = tf_vector(header)
		header_score = tf_idf_score(query,header_vector)
		### end header parse ###

		#parsing body string
		body_vector = tf_vector(body)
		body_score = tf_idf_score(query,body_vector)
		### end body parse ###

		#anchor tag weighted score
		if sum_of_weights>0:
			anchor_score = weighted_sum/sum_of_weights
		else:
			anchor_score = 0
		### end anchor tag score ###

		### relevance score ###
		query = query.replace("\n","")
		query = query.strip()
		query = str(query)
		try:
			relevance_score = rel_values[query][url_string]
		except:
			relevance_score = 0
		### end ###

		### creating record for each query-document pair ###
		record = []
		record.append(url_score)
		record.append(title_score)
		record.append(header_score)
		record.append(body_score)
		record.append(anchor_score)

		### adding to the training data vector ###
		train_data.append(record)
		relevance_data.append(relevance_score)

### building numpy array for training data ###
train_data_to_scale = np.array(train_data)

### scaling the training data for mean 0 and std.dev of 1 ###
train_data_scaled = preprocessing.scale(train_data_to_scale)

i = 0
j = 0
k = 0
### writing to output file
for record in train_data_scaled:
	j += 1
	new_training_data.write(str(record[0])+" | "+str(record[1])+" | "+str(record[2])+" | "+str(record[3])+" | "+str(record[4])+" | "+str(relevance_data[i])+" \n")
	#new_training_data_csv.write(str(record[0])+" , "+str(record[1])+" , "+str(record[2])+" , "+str(record[3])+" , "+str(record[4])+" , "+str(relevance_data[i])+" \n")
	if(query_result_count[k]==j):
		new_training_data.write("###query###\n")
		k += 1
		j = 0
	i += 1
print "Successfully generated training data\nquery-doc pair feature vectors are written into file\nCheck output folder .....\n"

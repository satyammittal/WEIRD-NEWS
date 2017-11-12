import pandas as pd 
import csv
import numpy as np
from sklearn.model_selection import train_test_split
import operator
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import json

def RandomforestTrain(train_x, train_y):
	clf = RandomForestClassifier()
	clf.fit(train_x, train_y)
	trained_model = clf
	return trained_model

def trainVectore():
	f=open("weirdnews.json","r")
	data = pd.read_csv('output.csv', low_memory=False)
	Y=data['LABEL']
	lines=[line for line in f][:500]
	titles=[json.loads(line)['title'] for line in lines] 
	docs=titles
	vectorizer = TfidfVectorizer(min_df=2,max_features= 650,analyzer='word',stop_words='english')
	# Train the vectorizer on the descriptions
	vectorizer = vectorizer.fit(docs)

def trainVectors():
	f=open("weirdnews.json","r")
	data = pd.read_csv('output.csv', low_memory=False)
	Y=data['LABEL']
	lines=[line for line in f][:500]
	titles=[json.loads(line)['title'] for line in lines] 
	docs=titles

	# got max on random forest at 650
	vectorizer = TfidfVectorizer(min_df=2,max_features= 1000,analyzer='word',stop_words='english')
	# Train the vectorizer on the descriptions
	vectorizer = vectorizer.fit(docs)
	# Convert descriptions to feature vectors
	X_tfidf = vectorizer.transform(docs)
	X=X_tfidf
	X=X.todense()
	X=X.tolist()
	return vectorizer,X,Y

def gettfidfVect(vectorizer,doc):
	X_tfidf = vectorizer.transform([doc])
	X=X_tfidf
	X=X.todense()
	X=X.tolist()
	return X;

def tfidf(X,Y):
	XTrain,XTest,YTrain,YTest= train_test_split(X,Y,test_size=0.2,random_state=7,stratify=Y)
	print "Using Random-Forest"
	trainedModel=RandomforestTrain(XTrain, YTrain)
	return trainedModel,XTrain;
	
def demo(title):
	vectorizer,X,Y=trainVectors()
	print len(X[0])
	trainedRF,XTrain=tfidf(X,Y)
	X=gettfidfVect(vectorizer,title)
	print len(X[0])
	predictions = trainedRF.predict(X)
	return predictions[0]
	# print "RUnning NN"
	# NNetwork(XTrain, YTrain, XTest, YTest)

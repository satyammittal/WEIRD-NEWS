import pandas as pd 
import csv
import numpy as np
from sklearn.model_selection import train_test_split
import operator
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import tree
import numpy
import keras
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense, LSTM
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC,LinearSVC
from sklearn.linear_model import LogisticRegression
from features import getFeature

def Linear_svm(train_x, train_y, test_x, test_y):
    svc = LinearSVC()
    svc =svc.fit(train_x, train_y)
    # Train and Test Accuracy
    return svc
    # prediction =  svc.predict(train_x)
    
def SVM_RBF(train_x,train_y):
    # SVM regularization parameter
    c=1.0
    rbf = SVC(kernel='linear', gamma=0.7, C=c)
    rbf = rbf.fit(train_x, train_y)
    # Train and Test Accuracy
    # prediction =  rbf.predict(train_x)
    return rbf;

def NeuralNetwork(XTrain, YTrain):
	model = Sequential()
	print np.shape(XTrain)
	print np.shape(YTrain)
	model.add(Dense(80, activation='relu', input_dim=np.shape(XTrain)[1]))
	model.add(Dropout(0.2))
	
	model.add(Dense(40, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(15, activation='relu'))
	model.add(Dropout(0.1))
	model.add(Dense(4, activation='softmax'))
	sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])

	x_train=np.array(XTrain)
	y_train=np.array(YTrain)
	
	y_train=y_train[:,np.newaxis]
	y_train=keras.utils.to_categorical(y_train, num_classes=4)
	
	model.fit(x_train, y_train,epochs=100,batch_size=128,shuffle=True,validation_split=0.25)
	return model;

def RandomforestTrain(train_x, train_y):
	clf = RandomForestClassifier()
	clf.fit(train_x, train_y)
	trained_model = clf
	return trained_model;

def trainVectors():
	f=open("weirdnews.json","r")
	data = pd.read_csv('output.csv', low_memory=False)
	Y=data['LABEL']
	lines=[line for line in f][:500]
	docs=[json.loads(line)['title'] for line in lines] 
	vectorizer = TfidfVectorizer(min_df=2,max_features= 1000,analyzer='word',stop_words='english')
	vectorizer = vectorizer.fit(docs)
	X_tfidf = vectorizer.transform(docs)
	X=X_tfidf.todense().tolist()
	return vectorizer,X,Y

def logistic_regression(train_x, train_y):
    """
    Training logistic regression model with train dataset features(train_x) and target(train_y)
    :param train_x:
    :param train_y:
    :return:
    """

    logistic_regression_model = LogisticRegression()
    logistic_regression_model.fit(train_x, train_y)
    trained_model = logistic_regression_model
    
    return trained_model

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
	return trainedModel,XTest,YTest;
	
def readFV():
	data = pd.read_csv('output.csv', low_memory=False)
	train, test = train_test_split(data, test_size=0.20, random_state=42)
	train_y = train['LABEL']
	train_x = train[['NOUN','VERB','STOPWORDS','WORDCOUNT','AVGWORDLENGHT','ELIPSIS','EXCLAMATION','QUESTION','COLON','QUOTES','NCV','WCV','NCN','WNN',"NW","WW","POSSESSIVENESS"]]
	test_y = test['LABEL']
	test_x  =  test[['NOUN','VERB','STOPWORDS','WORDCOUNT','AVGWORDLENGHT','ELIPSIS','EXCLAMATION','QUESTION','COLON','QUOTES','NCV','WCV','NCN','WNN',"NW","WW","POSSESSIVENESS"]]
	return	train_x,train_y
	
def demo(title):
	vectorizer,X,Y=trainVectors()
	print len(X[0])
	trainedRF,XTest,YTest=tfidf(X,Y)
	title="Son of the circus thrives as crude-oil traffic cop"
	pred=[]

	X_IP=gettfidfVect(vectorizer,title)
	print len(X[0])
	predictions = trainedRF.predict(X_IP)
	pred.append(predictions[0])
	print predictions


	NNModel=NeuralNetwork(X,Y)
	X_IP=np.array(X_IP)
	pred.append(np.argmax(NNModel.predict(X_IP)))
	
	X,Y=readFV()
	
	rbf=SVM_RBF(X,Y)
	print title
	X_IP=[getFeature(title)]
	
	print X_IP
	
	print "SVM :"
	print rbf.predict(X_IP)
	pred.append(rbf.predict(X_IP)[0])

	LRModel=logistic_regression(X,Y)
	print "LogisticRegression :"
	print LRModel.predict(X_IP)
	pred.append(LRModel.predict(X_IP)[0])

	NNModel=NeuralNetwork(X,Y)
	print "Neural Network :"
	X_IP=np.array(X_IP)
	print NNModel.predict(X_IP)
	pred.append(np.argmax(NNModel.predict(X_IP)))
	print "======================================================"
	print pred
	print round(float(sum(pred))/len(pred))
	# print title

demo("Two to get comfy coffin for overnight stay in Dracula's castle")
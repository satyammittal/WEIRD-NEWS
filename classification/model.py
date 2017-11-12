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
	
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
def randomforest(train_x, train_y, test_x, test_y):
	clf = RandomForestClassifier()
	clf.fit(train_x, train_y)
	trained_model = clf
	predictions = trained_model.predict(test_x)
	for i in xrange(0, 5):
		print "Actual outcome :: {} and Predicted outcome :: {}".format(list(test_y)[i], predictions[i])
    # Train and Test Accuracy
	print "Train Accuracy :: ", accuracy_score(train_y, trained_model.predict(train_x))
	print "Test Accuracy  :: ", accuracy_score(test_y, predictions)

def decisiontree(train_x, train_y, test_x, test_y):
	model = tree.DecisionTreeClassifier()
	model.fit(train_x, train_y)
	trained_model = model
	predictions = trained_model.predict(test_x)
	for i in xrange(0, 5):
		print "Actual outcome :: {} and Predicted outcome :: {}".format(list(test_y)[i], predictions[i])
    # Train and Test Accuracy
	print "Train Accuracy :: ", accuracy_score(train_y, trained_model.predict(train_x))
	print "Test Accuracy  :: ", accuracy_score(test_y, predictions)

def logistic_regression(train_x, train_y, test_x, test_y):
	"""
	Training logistic regression model with train dataset features(train_x) and target(train_y)
	:param train_x:
	:param train_y:
	:return:
	"""

	logistic_regression_model = LogisticRegression()
	logistic_regression_model.fit(train_x, train_y)
	trained_model = logistic_regression_model
	predictions = trained_model.predict(test_x)
	for i in xrange(0, 5):
		print "Actual outcome :: {} and Predicted outcome :: {}".format(list(test_y)[i], predictions[i])
    # Train and Test Accuracy
	print "Train Accuracy :: ", accuracy_score(train_y, trained_model.predict(train_x))
	print "Test Accuracy  :: ", accuracy_score(test_y, predictions)

def baseline_model():
	model = Sequential()
	model.add(Dense(1, input_dim=10, kernel_initializer='glorot_normal'))
	model.add(Activation('relu'))
	model.add(Dropout(0.2))
	model.add(Dense(64, init='normal'))
	model.add(Activation('relu'))
	model.add(Dropout(0.1))
	model.add(Dense(16, init='normal'))
	model.add(Activation('relu'))
	model.add(Dense(1, init='normal'))
	model.add(Activation('softmax'))
	model.compile(loss='mean_absolute_error', optimizer='adam')
	return model

def neural_network(train_x, train_y):
	encoder = LabelEncoder()
	encoder.fit(train_y)
	encoded_y = encoder.transform(train_y)
	estimator = KerasClassifier(build_fn=baseline_model, nb_epoch=100, batch_size=5, verbose=0)
	kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
	results = cross_val_score(estimator, train_x, encoded_y, cv=kfold)
	print("Results: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

def svm(XTrain, YTrain, XTest, YTest):
	svc = LinearSVC()
	svc.fit(XTrain, YTrain)
	print("accuracy on training data: {}".format(svc.score(XTrain, YTrain)))
	print("accuracy on test data: {}".format(svc.score(XTest, YTest)))

def NeuralNetwork(XTrain, YTrain, XTest, YTest):
	model = Sequential()
	print np.shape(XTrain)
	print np.shape(YTrain)
	model.add(Dense(250, activation='relu', input_dim=np.shape(XTrain)[1]))
	model.add(Dropout(0.2))
	
	# model.add(Dense(128, activation='relu'))
	# model.add(Dropout(0.5))
	model.add(Dense(90, activation='relu'))
	model.add(Dropout(0.1))
	model.add(Dense(2, activation='softmax'))
	sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])


	x_train=np.array(XTrain)
	x_test=np.array(XTest)
	y_train=np.array(YTrain)
	y_test=np.array(YTest)

	y_train=y_train[:,np.newaxis]
	y_test=y_test[:,np.newaxis]
	mapping = YTrain
	print "len mapping : ",len(mapping)
	print mapping


	y_train=keras.utils.to_categorical(y_train, num_classes=2)
	y_test=keras.utils.to_categorical(y_test, num_classes=2)


	print "len train set",np.shape(x_train),np.shape(y_train)
	print "len test set",np.shape(x_test),np.shape(y_test)
	model.fit(x_train, y_train,epochs=100,batch_size=128,shuffle=True,validation_split=0.25)
	acc= model.evaluate(x_test, y_test, batch_size=128)
	print acc

def tfidf():
	f=open("normalnews.json","r")
	Y=[]
	lines=[line for line in f] 
	titles=[json.loads(line)['title'] for line in lines] 

	f=open("weirdnews.json","r")
	lines2=[line for line in f] 
	titles.extend([json.loads(line)['title'] for line in lines2] )
	docs=titles
	Y=[1]*len(lines)
	Y.extend([0]*len(lines2))

	vectorizer = TfidfVectorizer(min_df=2,max_features= 1000,analyzer='word',stop_words='english')
	# Train the vectorizer on the descriptions
	vectorizer = vectorizer.fit(docs)
	# Convert descriptions to feature vectors
	X_tfidf = vectorizer.transform(docs)
	X=X_tfidf
	X=X.todense()
	X=X.tolist()
	print np.shape(X), np.shape(Y)
	XTrain,XTest,YTrain,YTest= train_test_split(X,Y,test_size=0.2,random_state=7,stratify=Y)
	print "Using Random-Forest"
	randomforest(XTrain, YTrain, XTest, YTest)

#tfidf()
data = pd.read_csv('FeatureVect(4).csv', low_memory=False)
train, test = train_test_split(data, test_size=0.25, random_state=42)
train_y = train['LABEL']
train_x = train[['NOUN','VERB','STOPWORDS','WORDCOUNT','AVGWORDLENGHT','ELIPSIS','EXCLAMATION','QUESTION','COLON','QUOTES','NCV','WCV','NCN','WNN']]
test_y = test['LABEL']
test_x = test[['NOUN','VERB','STOPWORDS','WORDCOUNT','AVGWORDLENGHT','ELIPSIS','EXCLAMATION','QUESTION','COLON','QUOTES','NCV','WCV','NCN','WNN']]
print "Using Random-Forest"
randomforest(train_x, train_y, test_x, test_y)
print "Decision-Tree"
decisiontree(train_x, train_y, test_x, test_y)
print "Logistic Regression"
logistic_regression(train_x, train_y, test_x, test_y)
print "SVM"
svm(train_x, train_y, test_x, test_y)
print "NN Model"
NeuralNetwork(train_x,train_y,test_x,test_y)

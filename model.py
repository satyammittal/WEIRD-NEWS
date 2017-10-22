import pandas as pd 
import csv
from sklearn.model_selection import train_test_split
import operator
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


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

data = pd.read_csv('FeatureVect.csv', low_memory=False)
train, test = train_test_split(data, test_size=0.2)
train_y = train['LABEL']
train_x = train[['NOUN','VERB','STOPWORDS','WORDCOUNT','AVGWORDLENGHT']]
test_y = test['LABEL']
test_x = test[['NOUN','VERB','STOPWORDS','WORDCOUNT','AVGWORDLENGHT']]
randomforest(train_x, train_y, test_x, test_y)
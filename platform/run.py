from flask import Flask, render_template, flash, request
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField
import pandas as pd 
import csv
import numpy as np
from sklearn.model_selection import train_test_split
import operator
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import tree
from features import getFeature 
from getPrediction import demo, trainVectors, tfidf, gettfidfVect
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
app = Flask(__name__)
# App config.
DEBUG = True
app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d442f47d441f27567d441f2b6176a'
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
class SearchForm(Form):
	name = TextField('Name:', validators=[validators.required()])

def randomforest(train_x, train_y, test_x, test_y):
	clf = RandomForestClassifier()
	clf.fit(train_x, train_y)
	global trained_model
	trained_model = clf
	predictions = trained_model.predict(test_x)
	for i in xrange(0, 5):
		print "Actual outcome :: {} and Predicted outcome :: {}".format(list(test_y)[i], predictions[i])
	# Train and Test Accuracy
	print "Train Accuracy :: ", accuracy_score(train_y, trained_model.predict(train_x))
	print "Test Accuracy  :: ", accuracy_score(test_y, predictions)

def before_app_request():
	data = pd.read_csv('FeatureVect(8).csv', low_memory=False)
	train, test = train_test_split(data, test_size=0.25, random_state=42)
	train_y = train['LABEL']
	#NOUN, VERB, STOPWORDS, NUMWORDS, AVG,ELIPSIS,EXC,QUES,COLON,QUOTES, NCV,WCV,NCN,WCN,NW,WW
	train_x = train[['NOUN','VERB','STOPWORDS','WORDCOUNT','AVGWORDLENGHT','ELIPSIS','EXCLAMATION','QUESTION','COLON','QUOTES','NCV','WCV','NCN','WNN','NW','WW']]
	test_y = test['LABEL']
	test_x = test[['NOUN','VERB','STOPWORDS','WORDCOUNT','AVGWORDLENGHT','ELIPSIS','EXCLAMATION','QUESTION','COLON','QUOTES','NCV','WCV','NCN','WNN','NW','WW']]
	print "Using Random-Forest"
	randomforest(train_x, train_y, test_x, test_y)
	global vectorizer, trainedRF
	vectorizer,X,Y=trainVectors()
	print len(X[0])
	trainedRF,XTrain=tfidf(X,Y)


@app.route("/", methods=['GET', 'POST'])
def hello():
	form = SearchForm(request.form)
 
	print form.errors
	if request.method == 'POST':
		title=request.form['name']
		print title
 
		if form.validate():
			# Save the comment here.
			feature = getFeature(title)
			weird = trained_model.predict([feature])
			print weird
			notify="Normal News"
			if weird[0] == 0 :
				X=gettfidfVect(vectorizer,title)
				print len(X[0])
				predictions = trainedRF.predict(X)
				percent = (predictions+1)*25
				notify= str(percent[0])+"% Weird News. "
			flash("\""+title +"\"" +' is ' + notify)
		else:
			flash('All the form fields are required. ')
 
	return render_template('home.html', form=form)

@app.route("/about", methods=['GET', 'POST'])
def about():
	return render_template('about.html')

def check_weird(title):
	feature = getFeature(title)
	weird = trained_model.predict([feature])
	print title
	print weird[0]
	if weird[0] == 0:
		return "Weird News"
	else:
		return "Normal News"

if __name__ == "__main__":
	before_app_request()
	app.run()

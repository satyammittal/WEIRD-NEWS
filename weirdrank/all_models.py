
# coding: utf-8

# In[ ]:


from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import numpy as np
# get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
import seaborn as sns; sns.set()  # for plot styling
from sklearn import cross_validation
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import numpy as np
# get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
import seaborn as sns; sns.set()  # for plot styling
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import classification_report

import csv
from sklearn.model_selection import train_test_split
import operator
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import tree
import numpy
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans

# import keras
# from keras.optimizers import SGD
# from keras.models import Sequential
# from keras.layers import Activation, Dropout, Flatten, Dense, LSTM
# from keras.wrappers.scikit_learn import KerasClassifier

import json
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC,LinearSVC
from sklearn.linear_model import LogisticRegression


# In[ ]:


seed = 7
numpy.random.seed(seed)
def randomforest(train_x, train_y, test_x, test_y):
    clf = RandomForestClassifier(n_estimators=50)
    clf = clf.fit(train_x, train_y)
    trained_model = clf
    predictions = trained_model.predict(test_x)
    for i in range(0, 5):
        print ("Actual outcome :: {} and Predicted outcome :: {}".format(list(test_y)[i], predictions[i]))
    # Train and Test Accuracy

    print ("Train Accuracy :: ", accuracy_score(train_y, trained_model.predict(train_x)))
    print( "Train confusion_matrix  :: \n", confusion_matrix(train_y, trained_model.predict(train_x)))
    print( "Train confusion_matrix  :: \n", classification_report(train_y, trained_model.predict(train_x)))
    
    print ("Test Accuracy  :: ", accuracy_score(test_y, predictions))
    print( "Test confusion_matrix  :: \n", confusion_matrix(test_y, predictions))
    print( "Test classification_report  :: \n", classification_report(test_y, predictions))
    
def decisiontree(train_x, train_y, test_x, test_y):
    model = tree.DecisionTreeClassifier()
    model.fit(train_x, train_y)
    trained_model = model
    predictions = trained_model.predict(test_x)
    for i in range(0, 5):
        print ("Actual outcome :: {} and Predicted outcome :: {}".format(list(test_y)[i], predictions[i]))
    # Train and Test Accuracy
    print ("Train Accuracy :: ", accuracy_score(train_y, trained_model.predict(train_x)))
    print( "Train confusion_matrix  :: \n", confusion_matrix(train_y, trained_model.predict(train_x)))
    print( "Train confusion_matrix  :: \n", classification_report(train_y, trained_model.predict(train_x)))
    
    print ("Test Accuracy  :: ", accuracy_score(test_y, predictions))
    print( "Test confusion_matrix  :: \n", confusion_matrix(test_y, predictions))
    print( "Test classification_report  :: \n", classification_report(test_y, predictions))


# In[ ]:

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
    for i in range(0, 5):
        print ("Actual outcome :: {} and Predicted outcome :: {}".format(list(test_y)[i], predictions[i]))
    # Train and Test Accuracy
    print ("Train Accuracy :: ", accuracy_score(train_y, trained_model.predict(train_x)))
    print( "Train confusion_matrix  :: \n", confusion_matrix(train_y, trained_model.predict(train_x)))
    print( "Train confusion_matrix  :: \n", classification_report(train_y, trained_model.predict(train_x)))
    
    print ("Test Accuracy  :: ", accuracy_score(test_y, predictions))
    print( "Test confusion_matrix  :: \n", confusion_matrix(test_y, predictions))
    print( "Test classification_report  :: \n", classification_report(test_y, predictions))


def Linear_svm(train_x, train_y, test_x, test_y):
    svc = LinearSVC()
    svc =svc.fit(train_x, train_y)
    # Train and Test Accuracy
    
    prediction =  svc.predict(train_x)
    print("accuracy on training data: {}".format(svc.score(train_x, train_y)))
    print( "Train confusion_matrix  :: \n", confusion_matrix(train_y, prediction))
    print( "Train confusion_matrix  :: \n", classification_report(train_y, prediction))
    
    prediction =  svc.predict(test_x)
    print("accuracy on test data: {}".format(svc.score(test_x, test_y)))
    print( "Test confusion_matrix  :: \n", confusion_matrix(test_y,prediction))
    print( "Test classification_report  :: \n", classification_report(test_y, prediction))
    
def SVM_RBF(train_x,train_y,test_x,test_y):
    # SVM regularization parameter
    c=1.0
    rbf = SVC(kernel='linear', gamma=0.7, C=c)
    rbf = rbf.fit(train_x, train_y)
    
    # Train and Test Accuracy
    prediction =  rbf.predict(train_x)
    print("accuracy on training data: {}".format(rbf.score(train_x, train_y)))
    print( "Train confusion_matrix  :: \n", confusion_matrix(train_y, prediction))
    print( "Train confusion_matrix  :: \n", classification_report(train_y, prediction))
    
    prediction =  rbf.predict(test_x)
    print("accuracy on test data: {}".format(rbf.score(test_x, test_y)))
    print( "Test confusion_matrix  :: \n", confusion_matrix(test_y,prediction))
    print( "Test classification_report  :: \n", classification_report(test_y, prediction))


# In[ ]:

print('Loading data ...')
# train = pd.read_csv('FeatureVect.csv')
# data = pd.read_csv('FeatureVect.csv', low_memory=False)
data = pd.read_csv('output.csv', low_memory=False)

train, test = train_test_split(data, test_size=0.2, random_state=42)

train_x = train.drop(['LABEL'], axis=1)
train_y = train['LABEL']

test_x = test.drop(['LABEL'], axis=1)
test_y = test['LABEL']


# In[ ]:


print ("\nUsing Random-Forest")
randomforest(train_x, train_y, test_x, test_y)


# In[ ]:

print ("\nUsing Decision-Tree")
decisiontree(train_x, train_y, test_x, test_y)


# In[ ]:

print ("\nUsing Logistic Regression")
logistic_regression(train_x, train_y, test_x, test_y)


# In[ ]:

print ("\nUsing Linear_svm")
Linear_svm(train_x, train_y, test_x, test_y)



# In[ ]:

def NeuralNetwork(XTrain, YTrain, XTest, YTest):
    model = Sequential()
#     print np.shape(XTrain)
#     print np.shape(YTrain)
    model.add(Dense(250, activation='relu', input_dim=np.shape(XTrain)[1]))
    model.add(Dropout(0.2))

    # model.add(Dense(128, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(90, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(4, activation='softmax'))
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])


    x_train=np.array(XTrain)
    x_test=np.array(XTest)
    y_train=np.array(YTrain)
    y_test=np.array(YTest)

    y_train=y_train[:,np.newaxis]
    y_test=y_test[:,np.newaxis]
    mapping = YTrain
    print ("len mapping : ",len(mapping))
    print (mapping)


    y_train=keras.utils.to_categorical(y_train, num_classes=4)
    y_test=keras.utils.to_categorical(y_test, num_classes=4)


    print ("len train set",np.shape(x_train),np.shape(y_train))
    print ("len test set",np.shape(x_test),np.shape(y_test))
    model.fit(x_train, y_train,epochs=100,batch_size=128,shuffle=True,validation_split=0.25)
    
    # Train and Test Accuracy
    acc= model.evaluate(x_train, y_train, batch_size=128)
    print ("Train Accuracy :: ", accuracy_score(train_y, trained_model.predict(train_x)))
    print( "Train confusion_matrix  :: \n", confusion_matrix(y_train, trained_model.predict(train_x)))
    print( "Train confusion_matrix  :: \n", classification_report(y_train, trained_model.predict(train_x)))
    
    acc= model.evaluate(x_test, y_test, batch_size=128)
    print ("Test Accuracy  :: ", acc)
    print( "Test confusion_matrix  :: \n", confusion_matrix(test_y, trained_model.predict(test_x)))
    print( "Test classification_report  :: \n", classification_report(test_y, trained_model.predict(test_x)))


# In[ ]:

# print ("NN Model")
# NeuralNetwork(train_x,train_y,test_x,test_y)


# In[ ]:

def tfidf():
    f=open("weirdnews.json","r")
    data = pd.read_csv('output.csv', low_memory=False)
    Y=data['LABEL']
    lines=[line for line in f][:500]
    titles=[json.loads(line)['title'] for line in lines] 
    docs=titles

    # got max on random forest at 650
    vectorizer = TfidfVectorizer(min_df=2,max_features= 650,analyzer='word',stop_words='english')
    # Train the vectorizer on the descriptions
    vectorizer = vectorizer.fit(docs)
    # Convert descriptions to feature vectors
    X_tfidf = vectorizer.transform(docs)
    X=X_tfidf
    X=X.todense()
    X=X.tolist()
    print (np.shape(X), np.shape(Y))
    XTrain,XTest,YTrain,YTest= train_test_split(X,Y,test_size=0.2,random_state=7,stratify=Y)
    print ("Using Random-Forest")
    randomforest(XTrain, YTrain, XTest, YTest)


# In[ ]:

print ("\nUsing tfidf")
tfidf()


# In[ ]:

print ("\nUsing SVM with RBF kernel")
SVM_RBF(train_x,train_y,test_x,test_y)


# In[ ]:




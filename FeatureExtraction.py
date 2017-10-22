'''
Created on Oct 21, 2017

@author: ethan
'''
import spacy
import re
import json
import pandas as pd
nlp=spacy.load('en')
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

stopwords=set(stopwords.words('english'))
def getFeatures(file,label):
    f=open(file,"r")
    lines=[line for line in f] 
    titles=[json.loads(line)['title'] for line in lines] 
    titlesTags = [[word.pos_ for word in nlp(line)] for line in titles]
    FV=[[
        tagseq.count('NOUN')+tagseq.count('PROPN'),
        tagseq.count('VERB'),
        len([word for word in re.sub('[^0-9a-zA-Z]+',' ',sent).split() if word.lower() in stopwords]),
        len(tagseq),
        sum([len(word) for word in re.sub('[^0-9a-zA-Z]+',' ',sent).split()])/len(re.sub('[^0-9a-zA-Z]+',' ',sent).split()),
        label
        ] for  tagseq,sent in zip(titlesTags,titles)]
    return FV


FV=getFeatures("normalnews.json", 1)
FV.extend(getFeatures("weirdnews.json", 0))
for item in FV:
    print item
df=pd.DataFrame(FV,columns=["NOUN","VERB","STOPWORDS","WORDCOUNT","AVGWORDLENGHT","LABEL"])
df.to_csv("FeatureVect2.csv",index=False,header=True,columns=["NOUN","VERB","STOPWORDS","WORDCOUNT","AVGWORDLENGHT","LABEL"])    
    
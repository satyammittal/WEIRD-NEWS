'''
Created on Oct 21, 2017

@author: ethan
'''
import spacy
import re
import json
import pandas as pd
import numpy as np
nlp=spacy.load('en')
from nltk.corpus import stopwords
from nltk.stem.porter import *
from collections import defaultdict
stemmer = PorterStemmer()
stopwords=set(stopwords.words('english'))

NNCommonVerbs=set([ "say", "get", "take", "kill", "make", "'s", "may", "call", "seek", "want", "give", "look", "hit", "meet", "could", "win", "go", "ask", "help", "use", "tri", "set", "turn", "arrest", "come", "face", "leav", "need", "see", "return", "show", "play", "put", "becom", "find", "move", "tell", "lead", "die", "rais", "pay", "know", "found", "plan", "rise", "warn", "expect", "lose", "accus", "talk"])
WNCommonVerbs=set([ "get", "found", "use", "find", "arrest", "take", "say", "make", "stolen", "seek", "offer", "give", "want", "may", "win", "turn", "rescu", "charg", "tri", "call", "sell", "lead", "return", "hit", "ban", "sue", "caught", "steal", "accus", "go", "save", "drive", "goe", "show", "run", "help", "'s", "pay", "miss", "live", "set", "stop", "eat", "kill", "lose", "leav", "keep", "face", "fall", "pull"])
NNCommonNouns=set([ "india", "modi", "u.s.", "new", "pm", "china", "pakistan", "us", "obama", "trump", "leader", "attack", "world", "case", "year", "bjp", "state", "presid", "court", "day", "time", "congress", "govern", "plan", "bank", "death", "report", "elect", "polic", "chief", "khan", "poll", "deal", "clinton", "govt", "man", "hous", "win", "minist", "iran", "game", "delhi", "parti", "war", "s", "test", "home", "team", "russia", "offici"])
WNCommonNouns=set([ "man", "polic", "woman", "dog", "car", "new", "school", "florida", "year", "home", "world", "babi", "citi", "student", "day", "sex", "coupl", "cat", "record", "u.s.", "driver", "teen", "hous", "zoo", "music", "bear", "video", "park", "york", "offic", "boy", "town", "news", "name", "famili", "store", "owner", "thief", "today", "california", "cop", "texa", "robber", "girl", "court", "men", "time", "snake", "women", "worker"])

pos_words = set(['i', 'we', 'you', 'he', 'she' ,'it', 'they', 'my', 'their', 'his', 'him', 'her', 'these', 'those', 'this', 'that'])
NW=set(['say', 'new', 'india', 'modi', 'kill', 'pm', 'us', 'china', 'year', 'obama', 'pakistan', 'attack', 'trump', 'win', 'get', 'leader', 'plan', 'world', 'take', 'first', 'state', 'may', 'case', 'bjp', 'report', 'make', 'presid', 'day', 'back', 'court', 'call', 'meet', 'indian', 'time', 'govern', 'congress', 'show', 'elect', 'chief', 'one', 'bank', 'deal', 'seek', 'help', 'open', 'top', 'death', 'govt', 'poll', 'polic'])
WW=set(['man', 'polic', 'woman', 'get', 'new', 'year', 'dog', 'car', 'arrest', 'found', 'use', 'school', 'home', 'florida', 'find', 'old', 'take', 'world', 'babi', 'say', 'british', 'charg', 'alleg', 'cat', 'citi', 'name', 'record', 'back', 'student', 'rescu', 'day', 'make', 'stolen', 'ban', 'coupl', 'suspect', 'sex', 'teen', 'call', 'fire', 'offer', 'win', 'bear', 'driver', 'park', 'show', 'seek', 'jail', 'music', 'hous'])

def url_feature(file1, cnt,url_dict):
    f1=open(file1,"r")
    lines=[line for line in f1] 
    titles1=[json.loads(line)['url'] for line in lines]
    
    vect=[]
    cnt=1
    for title in titles1:
        url=re.findall(r'.*:\/\/([^\/]*)\/.*',title)
        if url_dict[url[0]]==0:
            url_dict[url[0]]=cnt
            vect.append(cnt)
            cnt +=1
        else:
            vect.append(url_dict[url[0]])
            
    return vect, cnt,url_dict




def getFeatures(file,label,n_list):
    f=open(file,"r")
    lines=[line for line in f] 
    titles=[json.loads(line)['title'] for line in lines] 
    titlesTags = [[word.pos_ for word in nlp(line)] for line in titles]
    threeDot=[1 if "..." in text else 0 for text in titles]
    exc=[1 if "!" in text else 0 for text in titles]
    que=[1 if "?" in text else 0 for text in titles]
    colon=[1 if ":" in text else 0 for text in titles]  
    quotes=[1 if "\"" in text else 0 for text in titles]
    poss_w=[]
    for line in titles:
        flag=0
        for word in re.sub('[^0-9a-zA-Z]+',' ',line).split():
            if word in pos_words:
                flag=1
        poss_w.append(flag)
    FV=[[
        tagseq.count('NOUN')+tagseq.count('PROPN'),
        tagseq.count('VERB'),
        len([word for word in re.sub('[^0-9a-zA-Z]+',' ',sent).split() if word.lower() in stopwords]),
        len(tagseq),
        sum([len(word) for word in re.sub('[^0-9a-zA-Z]+',' ',sent).split()])/len(re.sub('[^0-9a-zA-Z]+',' ',sent).split()),
        td,e,q,c,qu,pw,
        len([word for word in re.sub('[^0-9a-zA-Z]+',' ',sent).split() if word in NNCommonVerbs or stemmer.stem(word) in NNCommonVerbs]),
        len([word for word in re.sub('[^0-9a-zA-Z]+',' ',sent).split() if word in WNCommonVerbs or stemmer.stem(word) in WNCommonVerbs]),
        len([word for word in re.sub('[^0-9a-zA-Z]+',' ',sent).split() if word in NNCommonNouns or stemmer.stem(word) in NNCommonVerbs]),
        len([word for word in re.sub('[^0-9a-zA-Z]+',' ',sent).split() if word in WNCommonNouns or stemmer.stem(word) in WNCommonVerbs]),
        len([word for word in re.sub('[^0-9a-zA-Z]+',' ',sent).split() if word in NW or stemmer.stem(word) in NW]),
        len([word for word in re.sub('[^0-9a-zA-Z]+',' ',sent).split() if word in WW or stemmer.stem(word) in WW]),        
        nl,
        label
        ] for  tagseq,sent,td,e,q,c,qu,nl,pw in zip(titlesTags,titles,threeDot,exc,que,colon,quotes,n_list,poss_w)]
    return FV

url_dict = defaultdict(int)
n_list,cnt,url_dict = url_feature("/home/ethan/eclipse-workspace/data/weirdnews/normalnews.json",1,url_dict)
w_list,cnt,url_dict = url_feature("/home/ethan/eclipse-workspace/data/weirdnews/weirdnews.json",cnt,url_dict)
print np.shape(w_list)
FV=getFeatures("/home/ethan/eclipse-workspace/data/weirdnews/normalnews.json", 1,n_list)
FV.extend(getFeatures("/home/ethan/eclipse-workspace/data/weirdnews/weirdnews.json", 0, w_list))
for item in FV:
    print item
df=pd.DataFrame(FV,columns=["NOUN","VERB","STOPWORDS","WORDCOUNT","AVGWORDLENGHT","ELIPSIS","EXCLAMATION","QUESTION","COLON","QUOTES","POSSESSIVENESS","NCV","WCV","NCN","WNN","NW","WW","DICT","LABEL"])
df.to_csv("FeatureVect.csv",index=False,header=True,columns=["NOUN","VERB","STOPWORDS","WORDCOUNT","AVGWORDLENGHT","ELIPSIS","EXCLAMATION","QUESTION","COLON","QUOTES","POSSESSIVENESS","NCV","WCV","NCN","WNN","NW","WW","DICT","LABEL"])    
    
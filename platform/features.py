'''
Created on Nov 11, 2017

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


def getFeature(title):
    pass
    threedot=0
    exc=0
    que=0
    colon=0  
    quotes=0
    poss_word=0
    for word in re.sub('[^0-9a-zA-Z]+',' ',title).split():
        if word in pos_words:
            poss_word=1
    
    if "..." in title:
        threedot=1
    if "!" in title:
        exc=1;
    if "?" in title:
        que=1
    if "\"" in title:
        quotes=1
    if ":" in title:
        colon=1
    
    tagseq=[word.pos_ for word in nlp(unicode(title))]
    sent=title
    
    FV=[
        tagseq.count('NOUN')+tagseq.count('PROPN'),
        tagseq.count('VERB'),
        len([word for word in re.sub('[^0-9a-zA-Z]+',' ',sent).split() if word.lower() in stopwords]),
        len([word for word in re.sub('[^0-9a-zA-Z]+',' ',sent).split()]),
#         len(tagseq),
        float(sum([len(word) for word in re.sub('[^0-9a-zA-Z]+',' ',sent).split()]))/len(re.sub('[^0-9a-zA-Z]+',' ',sent).split()),
        threedot,exc,que,colon,quotes,#poss_word,
        len([word for word in re.sub('[^0-9a-zA-Z]+',' ',sent).split() if word in NNCommonVerbs or stemmer.stem(word) in NNCommonVerbs]),
        len([word for word in re.sub('[^0-9a-zA-Z]+',' ',sent).split() if word in WNCommonVerbs or stemmer.stem(word) in WNCommonVerbs]),
        len([word for word in re.sub('[^0-9a-zA-Z]+',' ',sent).split() if word in NNCommonNouns or stemmer.stem(word) in NNCommonVerbs]),
        len([word for word in re.sub('[^0-9a-zA-Z]+',' ',sent).split() if word in WNCommonNouns or stemmer.stem(word) in WNCommonVerbs]),
        len([word for word in re.sub('[^0-9a-zA-Z]+',' ',sent).split() if word in NW or stemmer.stem(word) in NW]),
        len([word for word in re.sub('[^0-9a-zA-Z]+',' ',sent).split() if word in WW or stemmer.stem(word) in WW]),        
        ]
    return FV
    
    
#NOUN, VERB, STOPWORDS, NUMWORDS, AVG,ELIPSIS,EXC,QUES,COLON,QUOTES, NCV,WCV,NCN,WCN,NW,WW
    
    
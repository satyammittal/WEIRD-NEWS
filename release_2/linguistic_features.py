
# coding: utf-8

# In[ ]:

# linguistic features
from collections import defaultdict 
import re
import numpy as np
import json

from nltk.stem.porter import *

from nltk.corpus import stopwords


# In[ ]:

# find words dictionary for weird and normal news
stemmer = PorterStemmer()
stopwords=set(stopwords.words('english'))
def create_dict(file):
    f=open(file,"r")
    lines=[line for line in f] 
    titles=[json.loads(line)['title'] for line in lines] 
    word_dict= defaultdict(int)
    
    words = [item for title in titles for item in re.sub('[^a-zA-Z]+',' ',title).lower().split()]
    
#     word=[ stemmer.stem(word.lower()) for word in words if word.isalpha()]

    for word in words:
        print (word) 
        print (stemmer.stem(word.lower()))
        
    word = [ item for item in word if item not in stopwords]
    
    for w in word:
        if len(w) >= 2:
            word_dict[w] = word_dict[w] + 1
    # top 100 words
    words_list=[]
    
    for key in sorted(word_dict, key=word_dict.get, reverse=True):
        words_list.append(key)
    words_list=words_list[:100]
    return words_list


# In[ ]:

# n_dict = create_dict("normalnews.json")
# print( n_dict)
w_dict = create_dict("weirdnews.json")
print( w_dict)



# In[ ]:




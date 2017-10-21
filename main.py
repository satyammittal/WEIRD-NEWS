import pandas as ps
import json
import re
from utils import *
import pandas as pd

stopwords = []

def main():
	normal_news_file = 'normalnews.json'
	weird_news_file = 'weirdnews.json'
	file = 'data.json'
	num_words = []
	num_stop = []
	num_nouns = []
	num_verbs = []
	avg_word_length = []
	pos_tag = []
	link = []
	store_stopwords()
	with open(normal_news_file) as news:
		for line in news:
			normalnews = json.loads(line)
			title = re.sub('[^0-9a-zA-Z]+', ' ', normalnews['title'])
			num_words.append( number_words(normalnews['title']))
			num_stop.append( number_of_stopwords(title) )
			avg_word_length.append( average_word_len(title) )
			num_nouns.append( len(get_nouns(title)))
			num_verbs.append( len(get_verbs(title)))
			pos_tag.append( get_pos_counts(title) )
			link.append(normalnews['url'])
	data = pd.DataFrame({'url':link,'num_word':num_words,'avg_word_len':avg_word_length,'num_stop':num_stop,'num_nouns':num_nouns,'num_verbs':num_verbs})
	print data

def number_words(title):
	return len(title.split(' '))

def number_of_stopwords(title):
	words = title.split(' ')
	count = 0
	for word in words:
		for t in stopwords:
			if t == word:
				count += 1
	return count

def average_word_len(title):
	return len(title)/len(title.split(' '))

def store_stopwords():
	fd=open('stopwords','r')
	for stopword in fd:
		stopword=stopword.strip()
		if stopword:
			stopwords.append(stopword.strip())
  	fd.close()

main()

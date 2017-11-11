import json
import pdb
import numpy as np
import pickle as pkl
from ast import literal_eval
import json
from tqdm import tqdm
import keras
from keras.layers import Layer, Input, merge, Dense, LSTM, Bidirectional, GRU, SimpleRNN, Dropout
from keras.layers.merge import concatenate, dot, multiply, add
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from attention import AttentionWithContext


class RNN_Model:

    def __init__(self, normal_file, weird_file, title_max, word_embed_size):

        self.normal_file = normal_file
        self.weird_file = weird_file
        self.title_max = title_max
        self.word_embed_size = word_embed_size
        self.train = []
        self.train_truth = []
        self.test = []
        self.test_truth = []
        self.word_embed = json.load(open('glove_embed.json'))


    def create_model(self):

        title_words = Input(shape=(self.title_max, self.word_embed_size))
        
        #lstm_layer = LSTM(64, return_sequences=False)(title_words)
        lstm_layer = Bidirectional(LSTM(64, return_sequences=True))(title_words)
        attention_layer = AttentionWithContext()(lstm_layer)
        dropout = Dropout(0.25)(attention_layer)
        dense = Dense(64, activation='relu')(dropout)
        dense = Dense(32, activation='relu')(dropout)
        output = Dense(1, activation='sigmoid')(dense)

        self.model = Model(inputs=[title_words], outputs=output)
        self.model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])
 

def test_title(normal_file, weird_file, title_max, embed_size):
    model = RNN_Model(normal_file, weird_file, title_max, embed_size)
    model.create_model()
    model.model.load_weights('./weights25/weights-02-0.10.hdf5')
    title = raw_input('Type in the title\n')
    title = title.strip('\'').split()
    data = []
    temp = []
    for word in title[:title_max]:
        try:
            temp.append(model.word_embed[word.lower()])
        except:
            temp.append([0]*embed_size)
    for i in range(title_max - len(title)):
        temp.append([0]*embed_size)
    data.append(temp)
    data = np.asarray(data)
    np.reshape(data, (1,25,300))

    out = model.model.predict([data])
    if out[0] >= 0.5:
        print "1"
    else:
        print "0"

if  __name__ == '__main__':

    test_title('normalnews.json', 'weirdnews.json', 25, 300)

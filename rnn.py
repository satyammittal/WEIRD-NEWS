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


    def data_handler(self):
        
        normal_data = []
        weird_data = []
        with open(self.normal_file) as news:
            data_embed = []
            for line in tqdm(news):
                data = json.loads(line)
                title = data['title'].strip('\'').split()
                
                temp = []
                for word in title[:self.title_max]:
                    try:
                        temp.append(self.word_embed[word.lower()])
                    except:
                        temp.append([0]*self.word_embed_size)
                for i in range(self.title_max - len(title)):
                    temp.append([0]*self.word_embed_size)
                data_embed.append(temp)

            size = len(data_embed)
            self.train += data_embed[:int(size*0.8)]
            self.test += data_embed[int(size*0.8):]
            self.train_truth += [0]*(int(size*0.8))
            self.test_truth += [0]*(size-int(size*0.8))
        
        with open(self.weird_file) as news:
            data_embed = []
            for line in tqdm(news):
                data = json.loads(line)
                title = data['title'].strip('\'').split()
                
                temp = []
                for word in title[:self.title_max]:
                    try:
                        temp.append(self.word_embed[word.lower()])
                        print "fafa"
                    except:
                        temp.append([0]*self.word_embed_size)
                for i in range(self.title_max - len(title)):
                    temp.append([0]*self.word_embed_size)
                data_embed.append(temp)

            size = len(data_embed)
            self.train += data_embed[:int(size*0.8)]
            self.test += data_embed[int(size*0.8):]
            self.train_truth += [1]*(int(size*0.8))
            self.test_truth += [1]*(size-int(size*0.8))

        self.train = np.array(self.train)
        self.test = np.array(self.test)
        self.train_truth = np.array(self.train_truth)
        self.test_truth = np.array(self.test_truth)


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
        

    def fit_model(self, inputs, outputs, epochs):
        filepath="./weights/weights-{epoch:02d}-{val_loss:.2f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]
        self.model.fit(inputs, outputs, validation_split=0.2, epochs=epochs, callbacks=callbacks_list, verbose=1)

    
def train(normal_file, weird_file, title_max, embed_size):
    model = RNN_Model(normal_file, weird_file, title_max, embed_size)
    model.data_handler()
    model.create_model()
    model.model.summary()
    model.fit_model([model.train], model.train_truth, 15)


def test(normal_file, weird_file, title_max, embed_size):
    model = RNN_Model(normal_file, weird_file, title_max, embed_size)
    model.data_handler()
    model.create_model()
    model.model.summary()
    model.model.load_weights('./weights15/weights-05-0.12.hdf5')
    out = model.model.predict([model.test])

    hit = 0 
    for i in range(out.shape[0]):
        print out[i], model.test_truth[i]
        if out[i] >= 0.5 and model.test_truth[i] == 1:
            hit += 1
        elif out[i] <= 0.5 and model.test_truth[i] == 0:
            hit += 1

    print float(hit)/float(out.shape[0])


if  __name__ == '__main__':

    train('normalnews.json', 'weirdnews.json', 25, 300)
    #test('normalnews.json', 'weirdnews.json', 25, 300)

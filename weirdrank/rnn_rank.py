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
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


class RNN_Model:

    def __init__(self, filename, title_max, word_embed_size):

        self.filename = filename
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
        with open(self.filename) as news:
            data_embed = []
            truth = []
            for line in tqdm(news):
                data = json.loads(line)
                title = data['title'].strip('\'').split()
                a = [0, 0, 0, 0]
                a[data['rank']] = 1
                truth.append(a)
                
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
            self.train = data_embed[:int(size*0.8)]
            self.test = data_embed[int(size*0.8):]
            self.train_truth = truth[:int(size*0.8)]
            self.test_truth = truth[int(size*0.8):]
        
        self.train = np.array(self.train)
        self.test = np.array(self.test)
        self.train_truth = np.array(self.train_truth)
        self.test_truth = np.array(self.test_truth)


    def create_model(self):

        title_words = Input(shape=(self.title_max, self.word_embed_size))
        
        lstm_layer = Bidirectional(LSTM(64, return_sequences=True))(title_words)
        dropout = Dropout(0.3)(lstm_layer)
        attention_layer = AttentionWithContext()(dropout)
        dropout2 = Dropout(0.3)(attention_layer)
        dense = Dense(64, activation='relu')(dropout2)
        dropout3 = Dropout(0.3)(dense)
        #dense = Dense(32, activation='relu')(attention_layer)
        output = Dense(4, activation='sigmoid')(dropout3)

        self.model = Model(inputs=[title_words], outputs=output)
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['mse', 'accuracy'])
        

    def fit_model(self, inputs, outputs, epochs):
        filepath="./weights_multidrop_avg/weights-{epoch:02d}-{val_loss:.2f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]
        self.model.fit(inputs, outputs, validation_split=0.2, epochs=epochs, callbacks=callbacks_list, verbose=1)

    
def train(filename, title_max, embed_size):
    model = RNN_Model(filename, title_max, embed_size)
    model.data_handler()
    model.create_model()
    model.model.summary()
    model.fit_model([model.train], model.train_truth, 15)


def test(filename, title_max, embed_size):
    model = RNN_Model(filename, title_max, embed_size)
    model.data_handler()
    model.create_model()
    model.model.summary()
    model.model.load_weights('./weights_multidrop_avg/weights-08-0.70.hdf5')
    out = model.model.predict([model.test])

    train_out = model.model.predict([model.train])
    train_truth = []
    train_out2 = []
    for i in range(train_out.shape[0]):
        train_out2.append(np.argmax(train_out[i]))
        train_truth.append(np.argmax(model.train_truth[i]))


    out2 = []
    test_truth = []
    hit = 0 
    for i in range(out.shape[0]):
        final_class = np.argmax(out[i])
        out2.append(final_class)
        test_truth.append(np.argmax(model.test_truth[i]))
        if model.test_truth[i][final_class] == 1:
            hit += 1

    print float(hit)/float(out.shape[0])
    print model.model.evaluate([model.test], model.test_truth)

    print "Train confusion_matrix  :: \n", confusion_matrix(np.asarray(train_truth), np.asarray(train_out2))
    print "Train classification_report  :: \n", classification_report(np.asarray(train_truth), np.asarray(train_out2))
    
    print "Test confusion_matrix  :: \n", confusion_matrix(np.asarray(test_truth), np.asarray(out2))
    print "Test classification_report  :: \n", classification_report(np.asarray(test_truth), np.asarray(out2))

if  __name__ == '__main__':

    #train('average.json', 25, 300)
    test('average.json', 25, 300)

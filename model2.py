import pickle
import numpy as np
import random
import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Input, LSTM, Bidirectional, Dense, Embedding,Multiply,Dropout,concatenate
from keras import Model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from nltk.translate.bleu_score import corpus_bleu

ordinal_mappings = { 
        '':1, '--':2, 'Chc':3, 
        'Def':4, 'E':5, 'ENE':6, 'ESE':7, 
        'Lkly':8, 'N':9, 'NE':10, 'NNE':11, 'NNW':12, 
        'NW':13, 'S':14, 'SChc':15, 'SE':16, 'SSE':17, 
        'SSW':18, 'SW':19, 'W':20, 
        'WNW':21, 'WSW':22
    }

#############################

#Constants Required

EVENTS_DATASET_PATH = './dataset/all.events'
DESC_LABELS_PATH = './dataset/all.text'
TOKENIZER_PATH = './token2.pkl'
MODEL_PATH = './model2.h5'
RAW_TEST_DATASET_PATH = './rawtest2.pkl'
PROCESSED_TEST_DATASET_PATH = './test2.pkl'
HISTORY_PATH = './history.pkl'
MAX_COUNT = 29000
MAX_LEN_FEATURES = 113
MAX_LEN_DESC = 90
BATCH_SIZE = 2048
EPOCHS = 150
INPUT_VOCAB_SIZE = None
#############################

def save_file(obj_to_save,filepath):
    '''
    saves object to file
    '''
    f = open(filepath,'wb')
    pickle.dump(obj_to_save,f)
    f.close()
    print('Saved At ',filepath)

def preprocess_features(feature):
    '''
    creates a feature vector and returns it
    '''
    x = []
    attr = [ (f.split(':')[0],f.split(':')[1])  for f in feature.split()]
    for att in attr:
        #ignore mode bucket attribute for now
        if not 'mode-bucket' in att[0]:
            if 'time' in att[0]:
                time_vals = att[1].split('-')
                x.append(int(time_vals[0]))
                x.append(int(time_vals[1]))
            elif 'mode' in att[0]:
                x.append(ordinal_mappings[att[1]])
            else:
                x.append(int(att[1]))
    
    return x

def get_features(features,lines,tokenizer):
    '''
    creates feature vector as x1
    creates each time step description vector as x2
    creates true label for each time step as y
    '''
    
    vocab_size = len(tokenizer.word_index) + 1
    x1, x2, y =[],[],[]
    for feature,desc in zip(list(features),lines):
        seq = tokenizer.texts_to_sequences([desc])[0]
        for i in range(1, len(seq)):
            in_seq, out_seq = seq[:i], seq[i]
            in_seq = pad_sequences([in_seq], maxlen=MAX_LEN_DESC)[0]
            out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
            x1.append(feature)
            x2.append(in_seq)
            y.append(out_seq)
    x1,x2,y = np.array(x1),np.array(x2),np.array(y)

    print("x1 Shape ",x1.shape," x2 shape ",x2.shape," Y shape",y.shape)
    return x1,x2,y




        


def create_dataset():
    '''
    creates preprocessed dataset and returns it
    '''
    x = [] #will contain features
    y = [] #will contain labels
    f = open(EVENTS_DATASET_PATH,'r').read()
    features = f.split('\n') # ascii value of \n is 10
    f = open(DESC_LABELS_PATH,'r').read()
    labels = f.split('\n')
    for feature,label in zip(features,labels):
        try:
            pre_x = preprocess_features(feature)
            x.append(pre_x)
            y.append(label)
        except Exception as e:
            print(e)
            pass
    c = list(zip(x,y))
    random.shuffle(c)
    x[:],y[:] = zip(*c) #extract from zip
    x = pad_sequences(x,padding='post',maxlen=MAX_LEN_FEATURES)
    x = np.array(x)
    INPUT_VOCAB_SIZE = np.unique(x).shape[0]
    print("Input vocab size is ",INPUT_VOCAB_SIZE)
    print('X Shape  is ',x.shape,' Y len is ',len(y))
    lines = ['starttoken '+l+' endtoken' for l in y]
    lines = [l.replace('%','percent') for l in lines]
    print(lines[0])
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    save_file(tokenizer,TOKENIZER_PATH)
    x_train,y_train,x_val,y_val,x_test,y_test = split_dataset(x,y)
    save_file((x_test,y_test),RAW_TEST_DATASET_PATH)
    x1_train,x2_train,y_train =  get_features(x_train,y_train,tokenizer) #Training
    x1_val,x2_val,y_val =  get_features(x_val,y_val,tokenizer) #Validation dataset
    x1_test,x2_test,y_test =  get_features(x_test,y_test,tokenizer) #Testing dataset
    return x1_train,x2_train,y_train,x1_val,x2_val,y_val,x1_test,x2_test,y_test



def split_dataset(x,y,train_size = 0.8,val_size=0.05,test_size=0.15):
    '''
    splits the dataset in training,validation and testing datasets
    '''
    total = len(y)
    train_end = int(total * train_size)
    val_end = int(train_end + (total * val_size))
    return x[:train_end],y[:train_end],x[train_end:val_end],y[train_end:val_end],x[val_end:],y[val_end:]


def create_model(x1,x2,y):
    '''
    Defines the model and returns it

    '''
    input1 = Input(shape=(x1.shape[1],))
    input2 = Input(shape=(x2.shape[1],))
    emb = Embedding(y.shape[1],100,mask_zero=True)(input2)
    lstm1 = Bidirectional(LSTM(400,return_sequences=True))(emb)
    lstm2 = Bidirectional(LSTM(400,return_sequences=True))(lstm1)
    lstm3 = Bidirectional(LSTM(400))(lstm2)
    con = concatenate([lstm3,input1])
    dec1 = Dense(512,activation='relu')(con)
    dec2 = Dense(y.shape[1],activation='softmax')(dec1)
    model = Model(inputs=[input1, input2], outputs=dec2)
    return model


def calculate_bleu(predicted,actual):
    '''
    calculates bleu scores for test set
    '''
    hypothesis = []
    references = []
    for pred,act in zip(predicted,actual):
        act = [word for word in act.split() if word.isalnum()]
        references.append([act])
        hypothesis.append(pred.split())
    print(references[:2])
    print(hypothesis[:2])
    print('BLEU-1: %f' % corpus_bleu(references,hypothesis, weights=(1.0, 0, 0, 0)))
    print('BLEU-2: %f' % corpus_bleu(references,hypothesis,weights=(0.5, 0.5, 0, 0)))
    print('BLEU-3: %f' % corpus_bleu(references,hypothesis, weights=(0.3, 0.3, 0.3, 0)))
    print('BLEU-4: %f' % corpus_bleu(references,hypothesis,weights=(0.25, 0.25, 0.25, 0.25)))





x1_train,x2_train,y_train,x1_val,x2_val,y_val,x1_test,x2_test,y_test = create_dataset()
model = create_model(x1_train,x2_train,y_train)
print(model.summary())
save_file((x1_test,x2_test,y_test),PROCESSED_TEST_DATASET_PATH)

#model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
model.compile(
        optimizer=Adam(),
        loss='categorical_crossentropy',
        metrics=['acc'])
print(model.summary())
filename = MODEL_PATH
checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
history = model.fit([x1_train,x2_train],y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[checkpoint], verbose=1,validation_data=([x1_val,x2_val],y_val))
save_file((history.history),HISTORY_PATH)
print(model.evaluate([x1_test,x2_test],y_test))

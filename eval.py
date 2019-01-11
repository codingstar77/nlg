import pickle
import numpy as np
import random
import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Input, LSTM, Bidirectional, Dense, Embedding,Multiply,Dropout,concatenate,Multiply
from keras import Model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from nltk.translate.bleu_score import corpus_bleu
from keras.models import load_model



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
BATCH_SIZE = 1024
EPOCHS = 150
INPUT_VOCAB_SIZE = None
#############################

def calculate_bleu(predicted,actual):
    '''
    calculates bleu scores for test set
    '''
    hypothesis = []
    references = []
    for pred,act in zip(predicted,actual):
        act = [word for word in act.split()[1:-1] if word.isalpha()]
        pred = [word for word in pred.split() if word.isalpha()]
        references.append([act])
        hypothesis.append(pred)
    print(references[:2])
    print(hypothesis[:2])
    print('BLEU-1: %f' % corpus_bleu(references,hypothesis, weights=(1.0, 0, 0, 0)))
    print('BLEU-2: %f' % corpus_bleu(references,hypothesis,weights=(0.5, 0.5, 0, 0)))
    print('BLEU-3: %f' % corpus_bleu(references,hypothesis, weights=(0.3, 0.3, 0.3, 0)))
    print('BLEU-4: %f' % corpus_bleu(references,hypothesis,weights=(0.25, 0.25, 0.25, 0.25)))

    
def predict_desc(feature,tokenizer,model):
    reversed_dict = {v:k for k,v in tokenizer.word_index.items()}
    output = 'starttoken'
    feature = feature.reshape(1,MAX_LEN_FEATURES)
    for i in range(MAX_LEN_DESC):
        seq = tokenizer.texts_to_sequences([output])[0]
        seq = pad_sequences([seq],maxlen=MAX_LEN_DESC)
        pred = model.predict([feature,seq])
        pred = np.argmax(pred)
        pred = reversed_dict[pred]
        if pred == 'endtoken':
            break
        output += ' '+pred
    return " ".join(output.split()[1:])

TRAINED_MODEL_PATH = './test2/trained_model.h5'
TOKEN_PATH = './test2/token.pkl'
TEST_DATASET_PATH = './test2/rawtest.pkl'

testing_data = pickle.load(open(TEST_DATASET_PATH,'rb'))
tokenizer = pickle.load(open(TOKEN_PATH,'rb'))
model = load_model(TRAINED_MODEL_PATH)


x,y_true = testing_data

y_pred = []

for i in range(x.shape[0]):
  y_pred.append(predict_desc(x[i],tokenizer,model))


pickle.dump((y_pred,y_true),open('pred.pkl','wb'))
calculate_bleu(y_pred,y_true)
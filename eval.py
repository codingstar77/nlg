import numpy as np 
from tensorflow.keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import pickle
from nltk.translate.bleu_score import corpus_bleu

ordinal_mappings = { 
        '':1, '--':2, 'Chc':3, 
        'Def':4, 'E':5, 'ENE':6, 'ESE':7, 
        'Lkly':8, 'N':9, 'NE':10, 'NNE':11, 'NNW':12, 
        'NW':13, 'S':14, 'SChc':15, 'SE':16, 'SSE':17, 
        'SSW':18, 'SW':19, 'W':20, 
        'WNW':21, 'WSW':22
    }
time_mappings = {
    '13-21':1, '17-21':2, '17-26':3, '17-30':4, '21-30':5, '26-30':6, '6-13':7, '6-21':8, 
    '6-9':9, '9-21':10
}


EVENTS_DATASET_PATH = '../weather/all.events'
DESC_LABELS_PATH = '../weather/all.text'
TOKENIZER_PATH = './token.pkl'
MODEL_PATH = './model.pkl'
MAX_COUNT = 29000
MAX_LEN_FEATURES = 113
MAX_LEN_DESC = 90
BATCH_SIZE = 2048

def predict_desc(feature,tokenizer,model):
    reversed_dict = {v:k for k,v in tokenizer.word_index.items()}
    output = 'starttoken'
    feature = feature.reshape(MAX_LEN_FEATURES)
    for i in range(MAX_LEN_DESC):
        seq = tokenizer.texts_to_sequences([output])[0]
        seq = pad_sequences([seq],maxlen=MAX_LEN_DESC)
        pred = model.predict([feature,seq])
        pred = np.argmax(pred)
        pred = reversed_dict[pred]
        if pred == 'endtoken':
            break
        output += ' '+pred
    return output





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




TRAINED_MODEL_PATH = './test2/trained_model.h5'
TOKEN_PATH = './test2/token.pkl'
TEST_DATASET_PATH = './test2/rawtest.pkl'

testing_data = pickle.load(open(TEST_DATASET_PATH,'rb'))
tokenizer = pickle.load(open(TOKEN_PATH,'rb'))
model = load_model(TRAINED_MODEL_PATH)





for i in range(10):
    feature_vector = testing_data[0][1]
    pred = predict_desc(feature_vector,tokenizer,model)
    print("------------------------------")
    print("Input :",testing_data[0][i])
    print("\n")
    print("True Output :",testing_data[1][i])
    print("Predicted :",pred)
    







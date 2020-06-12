import nltk
from nltk.tokenize import RegexpTokenizer

from gensim.parsing.preprocessing import STOPWORDS
from gensim.models import Doc2Vec
from gensim.models import Word2Vec
from nltk.corpus import wordnet
from nltk.tokenize import RegexpTokenizer
import nltk
from scipy import spatial
import numpy as np
import sys
from itertools import groupby
import random
import logging
import gensim

class Word2vectUtils:
    def __init__(self, model, dim=300):
        self.not_found_word = 0
        self.dim=dim
        self.w2v_model = model

    def def_noword_zeros(self, word):
        self.not_found_word += 1
        return np.zeros(self.dim)

    def def_noword_random(self, word):
        self.not_found_word += 1
        np.random.seed(sum([ord( x )for x in word]))
        return np.random.rand(self.dim)

    def transform2Word2Vect(self, sentence, def_noword_function=def_noword_random):
        w2vect = []
        for i in range(len(sentence)):
            w2vect.append(self.w2v_model[sentence[i]] if sentence[i] in self.w2v_model else def_noword_function(self, sentence[i]) )
        return w2vect

    def getWord2VectModel(self):
        return self.w2v_model

PREPROCESS_STEPS = ['stop_words_removal']

def to_lowercase(data):
    return data.lower()

def tokenize(data):
    tokenizer = RegexpTokenizer(r'\w+')
    return tokenizer.tokenize(data)

def remove_stopwords(data):
    return [word for word in data if word not in STOPWORDS]

def data_preprocess(data, steps=[]):
    data = tokenize(data.lower())
    if 'stop_words_removal' in steps:
        data = remove_stopwords(data)
    return data

def qa_preprocessing(question, answers, steps):
    question = data_preprocess(question, steps)
    for sentence in answers:
        answer_list = tokenize(sentence.lower())
        answers_tag_list.append([word for word in answer_list if word not in STOPWORDS])
    return question, answers_tag_list

def getQaPairAsWord2Vect(qaPair, w2v, def_noword_function, MAX_WORDS=50):
    question = qaPair[0]
    answer = qaPair[1]
    q_vect = []
    a_vect = []
    for i in range(MAX_WORDS):
        q_vect.append( transform2Word2Vect(def_noword_function, question[i]) )
        a_vect.append( transform2Word2Vect(def_noword_function, answer[i]) )
    label = qaPair[2]
    return q_vect, a_vect, label

def word2vect_sum_representation(list1, list2, w2v_model, dim=300):
    sum_list1 = np.zeros(dim)
    sum_list2 = np.zeros(dim)
    mult_vector = np.ones(dim)
    for wq in list1:
        try:
            sum_list1 += w2v_model[wq]
        except Exception as e:
            logger.debug("Word not in word2vect vocabulary "+wq)
    for aq in list2:
        try:
            sum_list2 += w2v_model[aq]
        except Exception as e:
            logger.debug("Word not in word2vect vocabulary "+aq)
    return sum_list1, sum_list2

def word2vect_sum_representation(list1, list2, w2v_model):
    list1 = []
    list2 = []
    for wq in list1:
        try:
            list1 += [w2v_model[wq]]
        except Exception as e:
            logger.debug("Word not in word2vect vocabulary "+wq)
    for aq in list2:
        try:
            list2 += [w2v_model[aq]]
        except Exception as e:
            logger.debug("Word not in word2vect vocabulary "+aq)
    return list1, list2

"""
The average precision is precision averaged across all values of recall between 0 and 1
AvgP = (Sum(p(K)*r(k)))/(#relDocs) , where r(k) is an indicator
Precision = RelRetrieved/Retrieved
https://webcache.googleusercontent.com/search?q=cache:Y9HcueyIxXgJ:https://sanchom.wordpress.com/tag/average-precision/+&cd=5&hl=es&ct=clnk&gl=co
"""
def avg_precision(y_true, y_pred):
    zipped = zip(y_true, y_pred)
    zipped.sort(key=lambda x:x[1],reverse=True)
    np_y_true, np_y_pred = zip(*zipped)
    k_list = [i for i in range(len(np_y_true)) if int(np_y_true[i])==1]
    score = 0.
    r = np.sum(np_y_true).astype(np.int64)
    for k in k_list:
        Yk = np.sum(np_y_true[:k+1])
        score += Yk/float(k+1)
    if r==0:
        return 0
    score/=(r)
    return score


"""
https://en.wikipedia.org/wiki/Mean_reciprocal_rank
"""
def reciprocal_rank(y_true, y_pred):
    zipped = zip(y_true, y_pred)
    zipped.sort(key=lambda x:x[1],reverse=True)
    count_r = 1.0
    rr_score = 0.0
    for y_t,y_p in zipped:
        if(y_t!=1):
            count_r += 1
        else:
            rr_score = 1.0/count_r
            break
    if count_r-1==len(y_true):
        rr_score = 0.0
    return rr_score

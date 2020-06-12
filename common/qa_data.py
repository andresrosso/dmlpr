import numpy as np
import random
from os.path import exists
from os import makedirs
import os
from random import shuffle
import json
import nltk
import nltk.data
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

#WikiQA Imports
#import wikiqa_data_stats
#from wikiqa_helper import load_questions_from_file
#Trec Imports
#import trecqa_jakanahelper as trec

class QAPair():
    def __init__(self, qi, q, ai, a, l):
        self.qi = qi
        self.q = q
        self.ai = ai
        self.a = a
        self.l = l

    def __repr__(self):
        return 'qi('+str(self.qi)+') '+'ai('+str(self.ai)+')'+' '+str(self.l)

class QADataSet(object):

    def __init__(self, name):
        self.name = name
        self.patitions = []
        self.questions = {}

    def get_stats(self):
        raise NotImplementedError("Subclass must implement abstract method")

    def build_qa_pairs(self, dataset):
        raise NotImplementedError("Subclass must implement abstract method")


    def get_random_samples(self, dataset, samples, positive_rate=0.5):
        num_pos_samples = int(samples*(positive_rate))
        positiveSamples = [ q for q in dataset if q.l==1 ]
        negativeSamples = [ q for q in dataset if q.l==0 ]
        data = random.sample(positiveSamples, num_pos_samples)+random.sample(negativeSamples, samples-num_pos_samples)
        shuffle(data)
        return data


"""
class WikiQADataSet(QADataSet):

    def __init__(self):
        QADataSet.__init__(self,'WikiQA')
        self.patitions = ['train','validate','test']
        self.questions['train'], vocabulary, idf = load_questions_from_file('train', -1)
        self.questions['validate'], vocabulary, idf = load_questions_from_file('validate', -1)
        self.questions['test'], vocabulary, idf = load_questions_from_file('test', -1)

    def get_stats(self):
        return data_stats.getStats()

    '''
    Return a tuple of (quesion_id, question, answer_id, answer, label)
    '''
    def build_qa_pairs(self, dataset):
        #Construct Question Answer Pairs
        questions_answer_pairs = []
        for k, test_q_k in enumerate(dataset):
            q = test_q_k.question
            for i, a_i in enumerate(test_q_k.answers):
                is_correct = 1 if i in test_q_k.correct_answer else 0
                questions_answer_pairs += [QAPair(k+1, q, i, a_i, is_correct)]
        return questions_answer_pairs


class TrecDataSet(QADataSet):

    def __init__(self):
        QADataSet.__init__(self,'TrecDataSet')
        self.patitions = ['train','validate','test']
        self.questions['train'] = ( trec.load_data(trec.datasets['train']) )
        self.questions['validate'] = ( trec.load_data(trec.datasets['validate']) )
        self.questions['test'] = ( trec.load_data(trec.datasets['test']) )

    def get_stats(self):
        return 'Train: '+str(len(self.questions['train'][0]))+', Test: '+str(len(self.questions['test'][0]))+', Validate: '+str(len(self.questions['validate'][0]))

    '''CONCLUSION:
In otitis-prone young children, treating colds with this form of echinacea does not decrease the risk of acute otitis media, and may in fact increase risk. A regimen of up to five osteopathic manipulative treatments does not significantly decrease the risk of acute otitis media.
    Return a tuple of (quesion_id, question, answer_id, answer, label)
    '''
    def build_qa_pairs(self, dataset):
        #Construct Question Answer Pairs
        idx_ques, questions, idx_ans, answers, labels = dataset
        questions_answer_pairs = trec.buildQAPairs( idx_ques, questions, idx_ans, answers, labels )
        return questions_answer_pairs

class TrecDataSet_TrainAll(TrecDataSet):

    def __init__(self):
        TrecDataSet.__init__(self)
        self.name = 'TrecDataSet_TrainAll'
        self.questions['train'] = ( trec.load_data(trec.datasets['train-all']) )

"""

#BioASQ 2018
class BiosqDataSet(QADataSet):

    def __init__(self,year,path):
        QADataSet.__init__(self,'BiosqDataSet')
        print(year,path)
        questions = []
        self.question_files = []
        for f in os.listdir(path):
            self.question_files += [path+'/'+f]

    def get_stats(self):
        return 'Number of pairs: '+str(len(self.question_files))

    '''
    Return a tuple of (quesion_id, question, answer_id, answer, label)
    '''
    def build_qa_pairs(self, dataset):
        #Construct Question Answer Pairs
        self.questions_answer_pairs = []
        for f_q in self.question_files:
            data = json.load(open(f_q))
            index_ans = 0
            for ans in data['pos_answers']:
                index_ans += 1
                if(len(ans)>3 and len(data['question'])>3):
                    self.questions_answer_pairs += [QAPair(data['id'], data['question'], str(index_ans), ans, 1)]
            for ans in data['neg_answers']:
                index_ans += 1
                if(len(ans)>3 and len(data['question'])>3):
                    self.questions_answer_pairs += [QAPair(data['id'], data['question'], str(index_ans), ans, 0)]
        return self.questions_answer_pairs
    
#BioASQ 2019
class BioasqCuiTextDataSet(QADataSet):

    def __init__(self,year,path):
        QADataSet.__init__(self,'BioasqCuiTextDataSet')
        print(year,path)
        questions = []
        self.question_files = []
        for f in os.listdir(path):
            self.question_files += [path+'/'+f]

    def get_stats(self):
        return 'Number of pairs: '+str(len(self.question_files))

    '''
    Return a tuple of (quesion_id, question, quiestion_cui answer_id, answer, label)
    '''
    def build_qa_pairs(self, dataset):
        #Construct Question Answer Pairs
        self.questions_answer_pairs = []
        for f_q in self.question_files:
            data = json.load(open(f_q))
            for ans in data['pos_answers']:
                if(len(ans)>3 and len(data['question'])>3):
                    self.questions_answer_pairs += [QAPair(data['id'], (data['question'],data['question_cui']), 
                                                            str(ans['a_id']), (ans['a_t'],ans['a_cui']),1)]
            for ans in data['neg_answers']:
                if(len(ans)>3 and len(data['question'])>3):
                    self.questions_answer_pairs += [QAPair(data['id'], (data['question'],data['question_cui']), 
                                                            str(ans['a_id']), (ans['a_t'],ans['a_cui']),0)]
        return self.questions_answer_pairs

    
class DataSetFactory():
    @staticmethod
    def loadDataSet(targetclass, **kwargs):
        return globals()[targetclass](**kwargs)


def build_cuitext_pairs(q, q_id, doc, only_text=False):
    #[QAPair(data['id'], data['question'], str(index_ans), ans, 0)]
    q_list = []
    for i_s, sentence in enumerate(doc):
        if only_text:
            q_list += [QAPair(q_id, (q, []), str(q_id)+'_'+str(i_s),
                          (sentence,[]), 0)]
        else:
            q_list += [QAPair(q_id, (q, cui_extract_concepts(q)), str(q_id)+'_'+str(i_s),
                      (sentence,cui_extract_concepts(sentence)), 0)]
    return q_list


def build_pairs(q, q_id, doc):
    #[QAPair(data['id'], data['question'], str(index_ans), ans, 0)]
    q_list = []
    for i_s, sentence in enumerate(doc):
        q_list += [QAPair(q_id, q, str(q_id)+'_'+str(i_s), sentence, 0)]
    return q_list

def cui_extract_concepts(text, verbose=0):
    query = {"text": text }
    resp = requests.post('http://localhost:5000/match', json=query)
    cui_concepts = []
    if resp.status_code != 200:
        # This means something went wrong.
        raise ApiError('GET /tasks/ {}'.format(resp.status_code))
    for todo_item in resp.json():
        cui_concepts += [todo_item['cui']]
        if verbose > 0:
            print('{} {} {}'.format(todo_item['term'], todo_item['cui'], todo_item['similarity']))        
    return cui_concepts


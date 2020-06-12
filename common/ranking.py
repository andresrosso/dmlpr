from elasticsearch import Elasticsearch
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.tokenize import RegexpTokenizer
import numpy as np
import random
import json
import re
import bioasq_util

bioasq_util.es = Elasticsearch(hosts=['168.176.36.10:9200'])
index_name = '2018_pubmed_baseline_title_abs_mesh'
doc_relative_url = 'http://www.ncbi.nlm.nih.gov/pubmed/'
stops = stopwords.words('english')

def split_chunks(text):
    chunks = []
    new_sens = []
    if text is not None:
        words = re.findall('[A-Z]+[A-Z]+[A-Z]+:',text)
        answers = re.split('[A-Z]+[A-Z]+[A-Z]+:',text)
        sentences = []
        list_segments = list(map(lambda x: x[0]+x[1], zip(words, answers[1:])))
        if len(list_segments) > 0:
            for segment in list_segments:
                sentences.extend(sent_tokenize(segment))
        else:
            sentences = sent_tokenize(text)
        for i, sentence in enumerate(sentences):
            if text.find(sentence) < 0:
                print("--------->>>>>>"+sentence)
            chunks.append({'offsetInBeginSection': text.find(sentence),
                           'offsetInEndSection': text.find(sentence) + len(sentence),
                           'text': sentence})
    return chunks

"""
Pre-process the input data before representation
"""
def preprocess_text(text, max_lenght):
    tokens = word_tokenize(text.lower())
    return tokens[0:max_lenght]

"""
Simple word tokenizer
"""
def word_tokenize(data):
    tokenizer = RegexpTokenizer(r'\w+')
    return tokenizer.tokenize(data)

def rank_text(q, a):
    score = trained_model.predict(QAData.transform_sample(q,a))[0][0]
    return score

def rank_bioasq_passage(doc_id, question, passage, section):
    passage['beginSection'] = section
    passage['endSection'] = section
    passage['document'] = doc_relative_url + str(doc_id)
    passage['score'] = rank_text(question, passage['text'])
    return passage

def rank_document(question, doc_id, doc_title, doc_abstract):
    passages_ranked = []
    chunks_title = split_chunks(doc_title)
    title_passages_ranked = [ rank_bioasq_passage(doc_id, question, chunk, 'title') for chunk in chunks_title ]
    chunks_abstract = split_chunks(doc_abstract)
    abstract_passages_ranked = [ rank_bioasq_passage(doc_id, question, chunk, 'abstract') for chunk in chunks_abstract ]
    return title_passages_ranked + abstract_passages_ranked

def extract_rank_answer_candidates(question, docs):
    snippets = []
    for doc in docs:
        doc_id = doc.replace(doc_relative_url,'')
        doc_id, doc_title, doc_abstract = bioasq_util.get_doc(doc_id, index_name, remove_tags=True)
        snippets.extend(rank_document(question, doc_id, doc_title, doc_abstract))
    return snippets
from urllib.request import urlopen
import re
import socket
from elasticsearch import Elasticsearch
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from operator import itemgetter
import requests

es = Elasticsearch(hosts=['localhost:9200'])
stop_words = set(stopwords.words('english'))

def search_docs(question, index_name, num_docs):
    doc_list = []
    try:
        query = {
            "from" : 0, 
            "size" : 10,
            "query": {
                "multi_match" : {
                      "query":    "replace it!", 
                      "fields": [ "abstract", "title" ] 
                    }
            }
        }
        query['size'] = num_docs
        query['query']['multi_match']['query']=question
        res = es.search(index=index_name, body=query)
        
        for doc in res['hits']['hits']:
            doc_id = doc['_id']
            doc_title = doc['_source']['title']
            doc_abstract = doc['_source']['abstract']
            if remove_tags:
                doc_abstract = re.sub("<.*?>", "", doc_abstract)
            doc_list += [(doc_id, doc_title, doc_abstract)]
    except Exception as e:
        print("Error in query: ",question)
        print(e)
    return doc_list

def search_docs_mesh(question, index_name, num_docs, remove_tags=False):
    doc_list = []
    try:
        query = {
            "from" : 0, 
            "size" : num_docs,
            "query": {
                "multi_match" : {
                      "query":    question, 
                      "type":     "cross_fields",
                      "fields": [ "abstract", "title", "mesh" ],
                      "operator": "or"
                    }
            }
        }
        res = es.search(index=index_name, body=query, request_timeout=30)
        for doc in res['hits']['hits']:
            doc_id = doc['_id']
            doc_title = doc['_source']['title']
            doc_abstract = doc['_source']['abstract']
            doc_mesh = doc['_source']['mesh']
            if remove_tags:
                doc_abstract = re.sub("<.*?>", "", doc_abstract)
            doc_list += [(doc_id, doc_title, doc_abstract, doc_mesh)]
                
    except Exception as e:
        print("Error in query: ",question)
        print(e)
    return doc_list

def get_doc(doc_id, index_name, remove_tags=False):
    try:
        doc = None
        query = {
            "from" : 0, "size" : 1,
            "query": {
                "multi_match" : {
                      "query":    "replace it!", 
                      "fields": [ "_id" ] 
                    }
            }
        }
        query['query']['multi_match']['query']=doc_id
        res = es.search(index=index_name, body=query)
        for doc in res['hits']['hits']:
            doc_id = doc['_id']
            doc_title = doc['_source']['title']
            doc_abstract = doc['_source']['abstract']
            if remove_tags:
                doc_abstract = re.sub("<.*?>", "", doc_abstract)
            doc = (doc_id, doc_title, doc_abstract)
    except Exception as e:
        print("Error in query: ",doc_id)
        print(e)
    return doc


def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext
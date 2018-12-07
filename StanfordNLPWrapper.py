#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 10:36:26 2018

@author: phuongpham
"""

#### Wrapper for Stanford CoreNLP to use with Python
#### MODIFIED from https://www.khalidalnajjar.com/how-to-setup-and-use-stanford-corenlp-server-with-python/

from stanfordcorenlp import StanfordCoreNLP
#import logging
import json

class StanfordNLP:
    def __init__(self, host='http://localhost', port=9000):
        self.nlp = StanfordCoreNLP(host, port=port,
                                   timeout=30000)  # , quiet=False, logging_level=logging.DEBUG)
        self.props = {
#            'annotators': 'tokenize,ssplit,pos,lemma,ner,parse,depparse,dcoref,relation','sentiment'
            'annotators': 'sentiment',
            'pipelineLanguage': 'en',
            'outputFormat': 'json'
        }

    def word_tokenize(self, sentence):
        return self.nlp.word_tokenize(sentence)

    def pos(self, sentence):
        return self.nlp.pos_tag(sentence)

    def ner(self, sentence):
        return self.nlp.ner(sentence)

    def parse(self, sentence):
        return self.nlp.parse(sentence)

    def dependency_parse(self, sentence):
        return self.nlp.dependency_parse(sentence)

    def annotate(self, sentence):
#        return json.loads(self.nlp.annotate(sentence, properties=self.props))
        return self.nlp.annotate(sentence, properties=self.props)

    @staticmethod
    def tokens_to_dict(_tokens):
        tokens = defaultdict(dict)
        for token in _tokens:
            tokens[int(token['index'])] = {
                'word': token['word'],
                'lemma': token['lemma'],
                'pos': token['pos'],
                'ner': token['ner']
            }
        return tokens
    
def find_most_common(L):
    # L only contains values from 1-3, if there is a tight, return the higher value
    count = {3:0, 2:0, 1:0}
    for i in L:
        count[i] += 1
    
    max_num = None
    max_count = None
    for i in count:
        if max_count == None or count[i] > max_count:
            max_count = count[i]
            max_num = i
        elif count[i] == max_count:
            if i > max_num:
                max_num = i
    return max_num
       
# Global variable 
sNLP = StanfordNLP()  
def sentiment_analysis(text):
    # Take the description of the project and return a sentiment value
    # 0: Empty
    # 1: Negative
    # 2: Neutral
    # 3: Positive
    
    if len(text) == 0:
        return 0
    try:
        dict_anno = json.loads(sNLP.annotate(text))
        all_sentiment = []
        
        for i in dict_anno['sentences']:
            all_sentiment.append(int(i['sentimentValue']))
        return find_most_common(all_sentiment)
    except:
        return 0
    
    
        
### Sanity check
if __name__ == '__main__':
    sNLP = StanfordNLP()
    text = ''
    print(sentiment_analysis(text))

    
#    print("POS:", sNLP.pos(text))
#    print("Tokens:", sNLP.word_tokenize(text))
#    print("NER:", sNLP.ner(text))
#   print("Parse:", sNLP.parse(text))
#    print("Dep Parse:", sNLP.dependency_parse(text))
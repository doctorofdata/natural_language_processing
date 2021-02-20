#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 12:06:41 2021

@author: operator
"""

# Import 
import matplotlib.pyplot as plt
plt.xkcd()
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.model_selection import GridSearchCV
import gensim
import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel
from operator import itemgetter
from sklearn.manifold import TSNE
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer

stop_words = set(stopwords.words('english'))
words = set(nltk.corpus.words.words())
vader = SentimentIntensityAnalyzer()

# NLP
def process_txt(x):
    
    x = x.translate(str.maketrans('', '', string.punctuation))
    x = re.sub('\d+', '', x).lower().split()
    x = [i for i in x if i in words]
    x = [i for i in x if i not in stop_words and len(i) > 3]
    
    return x

def filter_pos(x):
    
    tags = nltk.pos_tag(x)
    x1 = [x[0] for x in tags if x[0] in words and len(x[0]) > 3 and x[1].startswith(('N', 'J', 'V'))]

    if len(x1) > 0:
        
        return ' '.join(x1)
    
    else:
        
        return None

def get_sentiment(x):
    
    return TextBlob(x).sentiment.polarity

def get_tone(score):
    
    if (score >= 0.1):
        
        label = "positive"
   
    elif (score <= -0.1):
        
        label = "negative"
        
    else:
        
        label = "neutral"
        
    return label
    
def build_lda(corpus, wordid, n):
    
    # Build optimized model
    model = gensim.models.ldamodel.LdaModel(corpus = corpus,
                                            id2word = wordid,
                                            num_topics = n,
                                            random_state = 100)
        
    # Get dominant topics
    topics = []

    for row in model[corpus]:
    
        topics.append(row)
    
    dom_topics = []

    for doc in topics:
    
        dom_topics.append(sorted(doc, key = lambda x: x[1], reverse = True)[0][0])
            
    return dom_topics

# Topic Modeling
class topic_model:
    
    def __init__(self, txt):
        
        self.txt = txt
        self.wordid = corpora.Dictionary(self.txt.apply(lambda x: x.split(' ')))
        self.corpus = [self.wordid.doc2bow(x.split(' ')) for x in self.txt]
        #self.scores = self.optimize_lda(20)
        #self.best = max(self.scores, key = itemgetter(1))[0]
        #self.t1, self.t2 = self.get_signals()
        
    def optimize_lda(self, nums):
        
        # Build topics using coherence
        scores = []

        for n in range(1, nums, 1):
    
            # Build optimized model
            model = gensim.models.ldamodel.LdaModel(corpus = self.corpus,
                                                    id2word = self.wordid,
                                                    num_topics = n,
                                                    random_state = 100)

            cm = CoherenceModel(model = model, corpus = self.corpus, coherence = 'u_mass')
            scores.append((n, cm.get_coherence()))
            
        return scores

    # Function to compute signals
    def get_signals(self):
        
        vec = CountVectorizer()
        out = vec.fit_transform(self.txt)
        tsne = TSNE(n_components = 2, random_state = 100).fit_transform(out)

        return tsne[:, 0], tsne[:, 1]
    
    # Function to get sentiment
    def get_emotional_content(self, x):
        
        return TextBlob(x).sentiment.polarity
    
    def the_dark_side(self, x):
        
        return vader.polarity_scores(x)['compound']
        
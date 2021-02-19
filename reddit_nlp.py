#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 11:50:38 2021

@author: operator
"""

# Import
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.model_selection import GridSearchCV

stop_words = set(stopwords.words('english'))
words = set(nltk.corpus.words.words())

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

# Get
df = pd.read_csv('/Users/operator/Documents/reddit_wsb.csv')

# Combine
df['text'] = df['title'].fillna(' ') + ' ' + df['body'].fillna(' ')

# Process
df['txt'] = df['text'].apply(process_txt)
df['pos'] = df['txt'].apply(filter_pos)

df1 = df[~df.pos.isna()]

# NMF Model
tfidf = TfidfVectorizer()
x = tfidf.fit_transform(df1['pos'])
fts = tfidf.get_feature_names()

# LDA 
cv = CountVectorizer()
vecs = cv.fit_transform(df1['pos'])
vecfts = cv.get_feature_names()

# Define Search Param
search_params = {'n_components': [5, 10, 15, 20, 25], 'learning_decay': [.5, .7, .9]}
lda = LatentDirichletAllocation()
model = GridSearchCV(lda, param_grid = search_params)
model.fit(vecs)


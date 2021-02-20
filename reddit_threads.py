#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 13:42:06 2021

@author: operator
"""

# Import 
import os
os.chdir('/Users/operator/Documents')
from nlpclass import *
from sklearn.feature_extraction.text import TfidfVectorizer
import networkx as nx
from gensim.summarization import keywords
import re

# Get
df = pd.read_csv('/Users/operator/Documents/reddit_wsb.csv')

# Combine
df['text'] = df['title'].fillna(' ') + ' ' + df['body'].fillna(' ')

# Process
df['txt'] = df['text'].apply(process_txt)
df['pos'] = df['txt'].apply(filter_pos)

df1 = df[~df.pos.isna()]

# LDA
topic_modeling = topic_model(df1['pos'])

# Build custom model
df1['topic'] = build_lda(topic_modeling.corpus, topic_modeling.wordid, 13)
df1['topic'].value_counts()

# Get tsne
t1, t2 = topic_modeling.get_signals()

# Visualize
fig, ax = plt.subplots(figsize = (10, 6))
ax.scatter(t1, t2, c = df1['topic'], s = .5)
plt.xlabel('tSNE 1')
plt.ylabel('tSNE 2')
plt.title('tSNE Scores for LDA Topics')

for nm, grp in df1.groupby('topic'):
    
    print(f'\nSamples for Topic {nm}: ')
    
    print(grp['text'].head(n = 10))
    
# # Get sentiment
# df1['sentiment'] = df1['pos'].apply(topic_modeling.the_dark_side)

# Visualize
fig, ax = plt.subplots(figsize = (10, 6))
ax.scatter(t1, t2, c = df1['topic'], s = df1['score']/100)
plt.xlabel('tSNE 1')
plt.ylabel('tSNE 2')
plt.title('tSNE Scores for LDA Topics')

# Insert tsne scores
df1['t1'] = t1
df1['t2'] = t2

# Get keywords
df1['keywords'] = df1['pos'].apply(lambda x: keywords(x))
df1['keywords'] = df1['keywords'].apply(lambda x: x.replace('\n', ' ').split('\''))
df1['keywords'] = [x[0].split(' ') for x in df1['keywords']]
df1['keywords'] = df1['keywords'].apply(lambda x: ' '.join(x))
#df2['keywords'] = df2['keywords'].apply(lambda x: " ".join(re.split("\s+", x, flags = re.UNICODE)))
df2 = df1.loc[df1['keywords'].astype(str).apply(lambda x: x.strip()) != '']

# LDA
topic_modeling = topic_model(df2['keywords'])
scores = topic_modeling.optimize_lda(20)

fig, ax = plt.subplots(figsize = (10, 6))
ax.plot([i[0] for i in scores], [i[1] for i in scores], lw = 1)
plt.xlabel('# Topics')
plt.ylabel('Coherence')
plt.title('Coherence Scores for Keywords LDA')

# Build custom model
df2['topic'] = build_lda(topic_modeling.corpus, topic_modeling.wordid, 10)
df2['topic'].value_counts()

df2.reset_index(drop = True, inplace = True)

# Get tsne
t1, t2 = topic_modeling.get_signals()

# Visualize
fig, ax = plt.subplots(figsize = (10, 6))
ax.scatter(t1, t2, c = df2['topic'], s = .5)
plt.xlabel('tSNE 1')
plt.ylabel('tSNE 2')
plt.title('tSNE Scores for LDA Topics')
plt.legend()

for nm, grp in df2.groupby('topic'):
    
    print(f'\nSamples for Topic {nm}: ')
    
    print(grp['keywords'].head(n = 10))
    
for nm, grp in df2.groupby('topic'):
    
    print(f'\nSamples for Topic {nm}: ')
    
    print(grp['text'].head(n = 10))
    

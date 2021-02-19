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
df1['topic'] = build_lda(topic_modeling.corpus, topic_modeling.wordid, 12)
df1['topic'].value_counts()

# Get tsne
t1, t2 = topic_modeling.get_signals()

# Visualize
fig, ax = plt.subplots(figsize = (10, 6))
ax.scatter(t1, t2, c = df1['topic'], s = 1)
plt.xlabel('tSNE 1')
plt.ylabel('tSNE 2')
plt.title('tSNE Scores for LDA Topics')

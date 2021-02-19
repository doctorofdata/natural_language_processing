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
import matplotlib.pyplot as plt
plt.xkcd()

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

# Visualize
fig, ax = plt.subplots(figsize = (10, 6))
ax.plot([i for i in range(1, 20)], topic_modeling.scores)
plt.xlabel('# Topics')
plt.ylabel('Coherence')
plt.title('Coherence Scores for LDA Topics')

# Build optimal model

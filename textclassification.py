#Importing the libraries we need:

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import base64
import string
import re
from collections import Counter
from nltk.corpus import stopwords
stopwords = stopwords.words('english')

df = pd.read_csv('7282_1.csv')
df.head()

#Show the fields we are going to use for the classification
df = df[['name','reviews.text']]
df.head()

#Show the number of rows and field
df.shape

#Show how many null values has the dataframe
df.isnull().sum()

#Import the necessary library to split the data to train and test sets
from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=0.33, random_state=46)
print('Reviews text sample:', train['reviews.text'].iloc[0])
print('Hotel reviewed:', train['name'].iloc[0])
print('Training Data Shape:', train.shape)
print('Testing Data Shape:', test.shape)

#Plotting the distribution of reviews by different hotels.
fig = plt.figure(figsize=(8,4))
sns.barplot(x = train['name'].unique(), y=train['name'].value_counts())
plt.show()

Importing the library to start text data preprocessing
import spacy

#We will try to find out the top words used in the reviews of two different hotels

nlp = spacy.load('en_core_web_sm')
punctuations = string.punctuation

def cleanup_text(docs, logging=False):
    texts = []
    counter = 1
    for doc in docs:
        if counter % 1000 == 0 and logging:
            print("Processed %d out of %d documents." % (counter, len(docs)))
        counter += 1
        doc = nlp(doc, disable=['parser', 'ner'])
        tokens = [tok.lemma_.lower().strip() for tok in doc if tok.lemma_ != '-PRON-']
        tokens = [tok for tok in tokens if tok not in stopwords and tok not in punctuations]
        tokens = ' '.join(tokens)
        texts.append(tokens)
    return pd.Series(texts)

QUALITY_text = [text for text in train[train['name'] == 'Quality Inn and Suites']['reviews.text']]

WESTERN_text = [text for text in train[train['name'] == 'Best Western Hospitality Hotel and Suites']['reviews.text']]

QUALITY_clean = cleanup_text(QUALITY_text)
QUALITY_clean = ' '.join(QUALITY_clean).split()

WESTERN_clean = cleanup_text(WESTERN_text)
WESTERN_clean = ' '.join(WESTERN_clean).split()

QUALITY_counts = Counter(QUALITY_clean)
WESTERN_counts = Counter(WESTERN_clean)

QUALITY_common_words = [word[0] for word in QUALITY_counts.most_common(20)]
QUALITY_common_counts = [word[1] for word in QUALITY_counts.most_common(20)]

fig = plt.figure(figsize=(18,6))
sns.barplot(x=QUALITY_common_words, y=QUALITY_common_counts)
plt.title('Most Common Words used in the reviews for hotel Quality Inn and Suites')
plt.show()

WESTERN_common_words = [word[0] for word in WESTERN_counts.most_common(20)]
WESTERN_common_counts = [word[1] for word in WESTERN_counts.most_common(20)]

fig = plt.figure(figsize=(18,6))
sns.barplot(x=WESTERN_common_words, y=WESTERN_common_counts)
plt.title('Most Common Words used in the reviews for hotel Best Western Hospitality Hotel and Suites')
plt.show()

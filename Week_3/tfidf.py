# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 16:50:39 2019

@author: COIATRIEL
"""

import pandas as pd
import numpy as np
np.random.seed(2019)

#load training data
bodies = pd.read_csv("train_bodies.csv")
headlines = pd.read_csv("train_stances.csv")

#clean data
import re
from gensim.parsing.preprocessing import remove_stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

def clean(doc):
    for i in range(doc.shape[0]):
        #lowercasing
        doc.set_value(i, doc.iloc[i].lower())
    #print("LOWERCASE")
    #print(doc)
    for i in range(doc.shape[0]):
        #remove punctuation
        doc.set_value(i, re.sub(r'([^\s\w])+', '', doc.iloc[i]))
    #print("REMOVE PUNCT")
    #print(doc)
    for i in range(doc.shape[0]):
        #remove stopwords
        doc.set_value(i, remove_stopwords(doc.iloc[i]))
    #print("REMOVE STOPWORDS")
    #print(doc)
    for i in range(doc.shape[0]):
        #tokenize
        doc.set_value(i, word_tokenize(doc.iloc[i]))
    #print("TOKENIZE")
    #print(doc)
    for i in range(doc.shape[0]):
        #lemmatize
        for j in range(len(doc.iloc[i])):
            doc.iloc[i][j] = lemmatizer.lemmatize(doc.iloc[i][j])
    #print("LEMMATIZE")
    #print(doc)
clean(bodies['articleBody'])
clean(headlines['Headline'])
train_data = pd.merge(bodies, headlines, on='Body ID')

#extract tfidf vectors of article body and headline
from sklearn.feature_extraction.text import TfidfVectorizer

def default(doc):
    return doc

def tfidf_extractor(b, h):
    data = pd.concat([b, h])
    vect = TfidfVectorizer(analyzer = 'word', preprocessor = default, tokenizer=default, token_pattern = None)
    fitted = vect.fit(data)
    return fitted

#calculate pairwise cosine distances - better than cosine similarity, because of the way sklearn handles cosine si
from sklearn.metrics.pairwise import paired_cosine_distances
def cosDistance(fitted, b, h):
    bodyTfidf = fitted.transform(b)
    headlineTfidf = fitted.transform(h)
    # print("Article Body")
    #print(bodyTfidf)
    #print("Headline")
    #print(headlineTfidf)
    cosDist = paired_cosine_distances(bodyTfidf, headlineTfidf)
    return cosDist

#SVM Classifier
from sklearn.linear_model import SGDClassifier
def svmClass(cd, target):
    svm = SGDClassifier()
    svm.fit(cd, target)
    return svm

#make predictions, report accuracy
def predict(classifier, testData, testTarget):
    predictions = classifier.predict(testData)
    return np.mean(predictions == testTarget)
    
#main action!
trainFitted = tfidf_extractor(train_data['articleBody'], train_data['Headline'])
trainCosDist = cosDistance(trainFitted, train_data['articleBody'], train_data['Headline'])
svmC = svmClass(trainCosDist.reshape(-1, 1), train_data['Stance'])

#load testing data
bodies = pd.read_csv("competition_test_bodies.csv")
headlines = pd.read_csv("competition_test_stances.csv")
clean(bodies['articleBody'])
clean(headlines['Headline'])
test_data = pd.merge(bodies, headlines, on='Body ID')


testFitted = tfidf_extractor(test_data['articleBody'], test_data['Headline'])
testCosDist = cosDistance(testFitted, test_data['articleBody'], test_data['Headline'])

print(predict(svmC, testCosDist.reshape(-1, 1), test_data['Stance']))
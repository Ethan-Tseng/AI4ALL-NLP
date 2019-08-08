import pandas as pd
import numpy as np
np.random.seed(2019)

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


#extract tfidf vectors of article body and headline
from sklearn.feature_extraction.text import TfidfVectorizer

def default(doc):
    return doc

def tfidf_extractor(b, h):
    data = pd.concat([b, h])
    vect = TfidfVectorizer(max_features = 2000, analyzer = 'word', preprocessor = default, tokenizer=default, token_pattern = None)
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

# Extract tf vector
def tfvec_extractor(b, h):
    data = pd.concat([b, h])
    vect = TfidfVectorizer(max_features = 500, use_idf = False, analyzer = 'word', preprocessor = default, tokenizer=default, token_pattern = None)
    fitted = vect.fit(data)
    return fitted

def tf_vectors(tf_fitted, b, h):
    bodyTf = tf_fitted.transform(b)
    headlineTf = tf_fitted.transform(h)
    return bodyTf, headlineTf

#SVM Classifier
from sklearn.linear_model import SGDClassifier
def svmClass(cd, target):
    svm = SGDClassifier()
    svm.fit(cd, target)
    return svm

#make predictions
def predict(classifier, testData, testTarget):
    predictions = classifier.predict(testData)
    return predictions


####################
# Perform Training #
####################
#load training data
bodies = pd.read_csv("train_bodies.csv")
headlines = pd.read_csv("train_stances.csv")
clean(bodies['articleBody'])
clean(headlines['Headline'])
train_data = pd.merge(bodies, headlines, on='Body ID')

# First feed in the combined article body and headline information and process jointly.
# This allows for extractors to know vocabulary words across entire corpus.
tfidf_Fitted = tfidf_extractor(train_data['articleBody'], train_data['Headline'])
tf_Fitted = tfvec_extractor(train_data['articleBody'], train_data['Headline'])

# Then get tf vectors and tfidf vectors for article bodies and headlines separately.
bodyTf, headlineTf = tf_vectors(tf_Fitted, train_data['articleBody'], train_data['Headline'])
tfidf_CosDist = cosDistance(tfidf_Fitted, train_data['articleBody'], train_data['Headline'])
tfidf_CosDist = tfidf_CosDist.reshape(-1,1)

# Combine into one vector
from scipy.sparse import hstack
stack = [bodyTf, headlineTf]
stack.extend([tfidf_CosDist for i in range(100)])
combined_input = hstack(stack).tocsc()

# Train SVM
svmC = svmClass(combined_input, train_data['Stance'])


###################
# Perform Testing #
###################
#load testing data
test_bodies = pd.read_csv("competition_test_bodies.csv")
test_headlines = pd.read_csv("competition_test_stances.csv")
clean(test_bodies['articleBody'])
clean(test_headlines['Headline'])
test_data = pd.merge(test_bodies, test_headlines, on='Body ID')


# First feed in the combined article body and headline information and process jointly.
# This allows for extractors to know vocabulary words across entire corpus.
test_tfidf_Fitted = tfidf_extractor(test_data['articleBody'], test_data['Headline'])
test_tf_Fitted = tfvec_extractor(test_data['articleBody'], test_data['Headline'])

# Then get tf vectors and tfidf vectors for article bodies and headlines separately.
test_bodyTf, test_headlineTf = tf_vectors(test_tf_Fitted, test_data['articleBody'], test_data['Headline'])
test_tfidf_CosDist = cosDistance(test_tfidf_Fitted, test_data['articleBody'], test_data['Headline'])
test_tfidf_CosDist = test_tfidf_CosDist.reshape(-1,1)

# Combine into one vector
test_stack = [test_bodyTf, test_headlineTf]
test_stack.extend([test_tfidf_CosDist for i in range(100)])
test_combined_input = hstack(test_stack).tocsc()

# Make predictions and evaluate
def test_predictions(test_data, predictions):
    stance_counts = {"unrelated":0, "discuss":0, "agree":0, "disagree":0}
    stance_correct_counts = {"unrelated":0, "discuss":0, "agree":0, "disagree":0}
    for i in range(test_data.shape[0]):
        example = test_data.iloc[i]
        predicted_stance = predictions[i]
        actual_stance = example['Stance']
        stance_counts[actual_stance] += 1
        if predicted_stance == actual_stance:
            stance_correct_counts[actual_stance] += 1
    return stance_counts, stance_correct_counts

predictions = predict(svmC, test_combined_input, test_data['Stance'])
stance_counts, stance_correct_counts = test_predictions(test_data, predictions)
print(stance_counts)
print(stance_correct_counts)


# FNC scoring as defined by the challenge
def FNC_score(test_data, predictions):
    score = 0.0
    perfect_score = 0.0
    for i in range(test_data.shape[0]):
        example = test_data.iloc[i]
        predicted_stance = predictions[i]
        actual_stance = example['Stance']
        if actual_stance == 'unrelated':
            perfect_score += 0.25
            if predicted_stance == 'unrelated':
                score += 0.25
        else: # related
            perfect_score += 1.0
            if predicted_stance == 'discuss' or predicted_stance == 'agree' or predicted_stance == 'disagree':
                score += 0.25
            if predicted_stance == actual_stance:
                score += 0.75
    return score / perfect_score
print("Final Score: {}".format(FNC_score(test_data, predictions)))
import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# from spacy.lang.en.stop_words import STOP_WORDS




if __name__ == '__main__':
    # read in data
    df = pd.read_csv('../data/lem-scripts')
    # train test split
    X = df['script']
    y = df['category']
    '''
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42,
        stratify=y)
    # instantiate vectorizer
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,3))
    # make tfidf of training and test text
    train_vectors = vectorizer.fit_transform(X_train).toarray()
    test_vectors = vectorizer.transform(X_test).toarray()
    # instantiate Multinomial Naive Bayes
    mnb = MultinomialNB()
    mnb.fit(train_vectors, y_train)
    # test predictions
    predictions = mnb.predict(test_vectors)
    # mean accuracy of test data and labels
    accuracy = mnb.score(test_vectors, y_test)
    '''

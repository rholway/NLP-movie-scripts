import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# from spacy.lang.en.stop_words import STOP_WORDS

def f(row):
    '''
    function to create two classes based on rotten tomatoes scores of above
    or below 75%
    '''
    if row['rating'] > 75:
        val = 1
    else:
        val = 0
    return val


if __name__ == '__main__':
    # read in data
    df = pd.read_csv('../data/partially-lem-df')
    df['r>75'] = df.apply(f, axis=1)
    df.drop(['Unnamed: 0', 'rating', 'category'], axis=1, inplace=True)
    # train test split
    X = df['script']
    y = df['r>75']

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

    # ROC plot
    fpr, tpr, thresholds = roc_curve(predictions, y_test)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('ROC Curve - Full Scripts', fontsize=20)
    plt.legend(loc="lower right", fontsize = 14)
    # plt.savefig('../images/roc-plot-lem-scripts')

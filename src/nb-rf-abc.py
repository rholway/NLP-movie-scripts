import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
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

def rf_score_plot(randforest, X_train, y_train, X_test, y_test):
    '''
        Parameters: randforest: RandomForestRegressor
                    X_train: 2d numpy array
                    y_train: 1d numpy array
                    X_test: 2d numpy array
                    y_test: 1d numpy array

        Returns: The prediction of a random forest regressor on the test set
    '''
    randforest.fit(X_train, y_train)
    y_test_pred = randforest.predict(X_test)
    test_score = mean_squared_error(y_test, y_test_pred)
    plt.axhline(test_score, alpha = 0.7, c = 'y', lw=3, ls='-.', label =
                                                        'Random Forest Test')


if __name__ == '__main__':
    # read in data
    df = pd.read_csv('../data/lem-scripts')
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
    # instantiate Random Forest Classifier
    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=1)
    rf.fit(train_vectors, y_train)
    # test predictions
    rf_predictions = rf.predict(test_vectors)
    # mean accuracy of test data and labels
    rf_accuracy = rf.score(test_vectors, y_test)

    # instantiate Multinomial Naive Bayes
    mnb = MultinomialNB()
    mnb.fit(train_vectors, y_train)
    # test predictions
    mnb_predictions = mnb.predict(test_vectors)
    # mean accuracy of test data and labels
    mnb_accuracy = mnb.score(test_vectors, y_test)

    # instantiat AdaBoostClassifier
    abr = AdaBoostClassifier(DecisionTreeClassifier(), learning_rate=0.1,
                                        n_estimators=100, random_state=1)
    abr.fit(train_vectors, y_train)
    # test predictions
    abr_predictions = mnb.predict(test_vectors)
    # mean accuracy of test data and labels
    abr_accuracy = mnb.score(test_vectors, y_test)




    # ROC plot
    rf_fpr, rf_tpr, rf_thresholds = roc_curve(rf_predictions, y_test)
    mnb_fpr, mnb_tpr, mnb_thresholds = roc_curve(mnb_predictions, y_test)
    abr_fpr, abr_tpr, abr_thresholds = roc_curve(abr_predictions, y_test)

    rf_roc_auc = auc(rf_fpr, rf_tpr)
    mnb_roc_auc = auc(mnb_fpr, mnb_tpr)
    abr_roc_auc = auc(abr_fpr, abr_tpr)

    plt.figure()

    plt.plot(rf_fpr, rf_tpr, color='darkorange', lw=1, label='Random Forest (area = %0.2f)' % rf_roc_auc)
    plt.plot(mnb_fpr, mnb_tpr, color='green', lw=1, label='MN Bayes (area = %0.2f)' % mnb_roc_auc)
    plt.plot(abr_fpr, abr_tpr, color='blue', lw=1, label='AdaBoost (area = %0.2f)' % abr_roc_auc)

    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('ROC Curve - Lemmatized Scripts', fontsize=20)
    plt.legend(loc="lower right", fontsize = 10)
    plt.savefig('../images/roc-plot-lemmy-scripts')

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


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
    # df = pd.read_csv('../data/scripts-rating-df')
    # df = pd.read_csv('../data/partially-lem-df')
    # df = pd.read_csv('../data/lem-scripts')
    # df = pd.read_csv('../data/lem-df-budgets')
    df = pd.read_csv('../data/partially-lem-df-budgets1')

    df['r>75'] = df.apply(f, axis=1)
    df.drop(['Unnamed: 0', 'rating', 'category'], axis=1, inplace=True)
    # train test split
    # X = df['script']
    X = df[['script', 'budget']]
    y = df['r>75']


    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42,
        stratify=y)
    # instantiate vectorizer
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,3))

    # make tfidf of training and test text
    # train_vectors = vectorizer.fit_transform(X_train)
    train_vectors = vectorizer.fit_transform(X_train['script'])
    train_vectors = np.hstack((np.expand_dims(X_train['budget'], axis=1), train_vectors.todense()))

    # test_vectors = vectorizer.transform(X_test)
    test_vectors = vectorizer.transform(X_test['script'])
    test_vectors = np.hstack((np.expand_dims(X_test['budget'], axis=1), test_vectors.todense()))

    model = Sequential()
    model.add(Dense(64, input_dim=10001, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    model.fit(train_vectors, y_train,
              epochs=20,
              batch_size=128)
    score = model.evaluate(test_vectors, y_test, batch_size=128)

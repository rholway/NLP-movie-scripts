import matplotlib.pyplot as plt
from sklearn import decomposition, preprocessing
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def f(row):
    '''
    function to create two classes based on rotten tomatoes scores of above
    or below 75%
    '''
    if row['rating'] > 75:
        val = 2
    else:
        val = 1
    return val

def plot_embedding(X, y, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    plt.figure(figsize=(5, 5), dpi=250)
    ax = plt.subplot(111)
    ax.axis('off')
    ax.patch.set_visible(False)
    ax.set_title('PCA of Full Scripts - Two Dimensions')
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(full_df['r>75'][i]), color=plt.cm.Set1(y[i] / 10),
                 fontdict={'weight': 'bold', 'size': 12})
    #


    plt.xticks([]), plt.yticks([])
    plt.ylim([-0.1,1.1])
    plt.xlim([-0.1,1.1])

    if title is not None:
        plt.title(title, fontsize=16)

if __name__ == '__main__':
    full_df = pd.read_csv('../data/scripts-rating-df')
    full_df['r>75'] = full_df.apply(f, axis=1)
    # lem_df = pd.read_csv('../data/lem-scripts')
    # lem_df['r>75'] = lem_df.apply(f, axis=1)

    X = full_df['script']
    y = full_df['r>75']


    tfidf_vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,3))
    full_tfidf = tfidf_vectorizer.fit_transform(X).todense()

    pca = decomposition.PCA(n_components=2)
    X_pca = pca.fit_transform(full_tfidf)
    plot_embedding(X_pca, full_df['r>75'])
    plt.savefig('../images/pca-full-scripts.png')

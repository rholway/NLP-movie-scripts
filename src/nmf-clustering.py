from sklearn.decomposition import NMF
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

def get_len(str):
    return len(str)

def fit_nmf(k):
    nmf = NMF(n_components=k)
    nmf.fit(full_tfidf)
    W = nmf.transform(full_tfidf);
    H = nmf.components_;
    return nmf.reconstruction_err_

def top_tokens_in_topic(topic_n, n_tokens, H):
    '''
    input: topic to explore (int), how many tokens to return (int)
    output: top tokens for topic_n
    '''
    top_movies = H.iloc[topic_n].sort_values(ascending=False).index[:n_tokens]
    return top_movies

def top_movies_in_topic(topic_n, n_movies, W):
    top_movies = W.iloc[:,topic_n].sort_values(ascending=False).index[:n_movies]
    return df['title'][top_movies]

def get_category_of_movies_in_topic(topic_n, n_categories, W):
    categories = W.iloc[:,topic_n].sort_values(ascending=False).index[:n_categories]
    return df['category'][categories]

if __name__ == '__main__':
    # read in df
    df = pd.read_csv('../data/lem-scripts')
    df.drop('Unnamed: 0', axis=1, inplace=True)

    # NMF on tfidf
    X = df['script']
    tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,3))
    k = 5 # number of topics
    topics = ['latent_topic_{}'.format(i) for i in range(k)]
    nmf = NMF(n_components = k)

    # full tfidf scripts
    full_tfidf = tfidf_vectorizer.fit_transform(X).todense()
    full_tfidf_feature_names = tfidf_vectorizer.get_feature_names()
    full_titles = X.index
    nmf.fit(full_tfidf)
    W = nmf.transform(full_tfidf)
    H = nmf.components_
    W = pd.DataFrame(W, index = full_titles, columns = topics)
    H = pd.DataFrame(H, index = topics, columns = full_tfidf_feature_names)
    W, H = (np.around(x,2) for x in (W, H))

    # elbow plot of NMF from plots
    error = [fit_nmf(i) for i in range(1,10)]
    plt.plot(range(1,10), error)
    plt.xlabel('k')
    plt.ylabel('Reconstruction Error')
    # plt.show()

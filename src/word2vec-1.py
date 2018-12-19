import pandas as pd
from gensim.models import Word2Vec
from gensim import corpora, models, similarities
from sklearn import cluster
from sklearn import metrics
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np


'''
def elbow_plot(data):
    plt.clf()
    ks = np.arange(2,20,2)
    sses = []
    for k in ks:
        model = cluster.KMeans(n_clusters = k, verbose = True)
        model.fit(data)
        sses.append(model.score(data))
    plt.plot(ks, sses)
    plt.xlabel('Number of clusters')
    plt.ylabel('SSE')
    plt.title('Elbow Plot')
    plt.show()
'''
if __name__ == '__main__':
    df = pd.read_csv('../data/partially-lem-df')
    scripts_list = df['script'].tolist()
    word_list = [script.split() for script in scripts_list]

    model = Word2Vec(word_list, min_count=1, size=100, sg=1)
    X = model[model.wv.vocab]
    vocab = list(model.wv.vocab.keys())


    kmeans = cluster.KMeans(n_clusters=6)
    kmeans.fit(X)

    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    # Print out the centroids.
    print("\n cluster centers:")
    print(centroids)

    # Find the top 10 features for each cluster.
    top_centroids = centroids.argsort()[:,-1:-51:-1]
    print("\n top features (words) for each cluster:")
    for num, centroid in enumerate(top_centroids):
        print("%d: %s" % (num, ", ".join(vocab[i] for i in centroid)))

    # elbow_plot(X)

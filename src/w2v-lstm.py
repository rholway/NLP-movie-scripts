import numpy as np
import gensim
import string
import pandas as pd

from keras.callbacks import LambdaCallback
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.layers import Dense, Activation
from keras.models import Sequential

def word2idx(word):
  return word_model.wv.vocab[word].index
def idx2word(idx):
  return word_model.wv.index2word[idx]

if __name__ == '__main__':
    df = pd.read_csv('../data/partially-lem-df')
    scripts = df['script'].tolist()
    # word_list = [script.split() for script in scripts]
    trial = ['in',
     'science',
     'and',
     'engineering,',
     'intelligent',
     'processing',
     'of',
     'complex',
     'signals',
     'such']

    print('\nTraining word2vec...')
    word_model = gensim.models.Word2Vec([trial], size=100, min_count=1, window=1, iter=100)
    pretrained_weights = word_model.wv.syn0
    vocab_size, emdedding_size = pretrained_weights.shape

    print('Result embedding shape:', pretrained_weights.shape)
    print('Checking similar words:')

    for word in ['in', 'science', 'complex', 'signals']:
        most_similar = ', '.join('%s (%.2f)' % (similar, dist) for similar, dist in word_model.most_similar(word)[:8])
        print(f'  {word} -> {most_similar}')

    print('\nPreparing the data for LSTM...')
    train_x = np.zeros([len(trial), 40], dtype=np.int32)
    train_y = np.zeros([len(trial)], dtype=np.int32)
    for i, script in enumerate(trial):
        for t, word in enumerate(script[:-1]):
            train_x[i, t] = word2idx(word)
        train_y[i] = word2idx(script[-1])
    print('train_x shape:', train_x.shape)
    print('train_y shape:', train_y.shape)

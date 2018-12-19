import numpy as np
import pandas as pd
import gensim
import string

from keras.callbacks import LambdaCallback
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.layers import Dense, Activation
from keras.models import Sequential
from keras.utils.data_utils import get_file

max_script_len = 1000

# df = pd.read_csv('../data/w2v-lstm.py')
df = pd.read_csv('../data/for-embedding-df')

scripts = df['script'].tolist()
trial = scripts


list_of_scripts = [[word for word in scr.lower().translate(string.punctuation).split()
        [:max_script_len]] for scr in trial]

print('Num scripts:', len(list_of_scripts))

print('\nTraining word2vec...')
# Here we instantiate the model. Can edit parameters in Word2Vec below
word_model = gensim.models.Word2Vec(list_of_scripts, size=100, min_count=1, window=2, iter=100)
# Pretrained weights are a list of vectors, where each vector represents a word
# Each vector has a length of 'size' from the word model above
pretrained_weights = word_model.wv.syn0
# vocab size is all of words in vocabulary
# embedding size is size of one word vector ('size' from word model)
vocab_size, embedding_size = pretrained_weights.shape
print('Result embedding shape:', pretrained_weights.shape)
print('Checking similar words:')
# Pick a few words, and find the eight most similar words.  Can change the # in the index
for word in [ 'christmas', 'winter', 'cake', 'holiday', 'mountain', 'sport', 'run',
                'talk', 'like', 'turn', 'danger', 'dog']:
  most_similar = ', '.join('%s (%.2f)' % (similar, dist) for similar, dist in word_model.most_similar(word)[:8])
  print('  %s -> %s' % (word, most_similar))

# find the index of a given word from the vocab
def word2idx(word):
  return word_model.wv.vocab[word].index
# find the word of a given index from the vocab
def idx2word(idx):
  return word_model.wv.index2word[idx]

print('\nPreparing the data for LSTM...')
train_x = np.zeros([len(list_of_scripts), max_script_len], dtype=np.int32)
train_y = np.zeros([len(list_of_scripts)], dtype=np.int32)
for i, script in enumerate(list_of_scripts):
  for t, word in enumerate(script[:-1]):
    train_x[i, t] = word2idx(word)
  train_y[i] = word2idx(script[-1])
print('train_x shape:', train_x.shape)
print('train_y shape:', train_y.shape)

print('\nTraining LSTM...')
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_size, weights=[pretrained_weights]))
model.add(LSTM(units=embedding_size))
model.add(Dense(units=vocab_size))
model.add(Activation('softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

def sample(preds, temperature=1.0):
  if temperature <= 0:
    return np.argmax(preds)
  preds = np.asarray(preds).astype('float64')
  preds = np.log(preds) / temperature
  exp_preds = np.exp(preds)
  preds = exp_preds / np.sum(exp_preds)
  probas = np.random.multinomial(1, preds, 1)
  return np.argmax(probas)

def generate_next(text, num_generated=100):
  word_idxs = [word2idx(word) for word in text.lower().split()]
  for i in range(num_generated):
    prediction = model.predict(x=np.array(word_idxs))
    idx = sample(prediction[-1], temperature=0.7)
    word_idxs.append(idx)
  return ' '.join(idx2word(idx) for idx in word_idxs)

def on_epoch_end(epoch, _):
  print('\nGenerating text after epoch: %d' % epoch)
  texts = [
    'it',
    'the',
    'we',
    'love',
    'what',
    'how',
    'when',
    'if',
    'however',
    'who',
    'where',
    'get'
  ]
  for text in texts:
    sample = generate_next(text)
    print('%s... -> %s' % (text, sample))

model.fit(train_x, train_y,
          batch_size=128,
          epochs=20,
          callbacks=[LambdaCallback(on_epoch_end=on_epoch_end)])

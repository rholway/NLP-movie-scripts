import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer



if __name__ == '__main__':
    df1 = pd.read_csv('../data/lem-scripts')
    df = df1.iloc[[117, 850, 315, 764, 944]].reset_index()


    vectorizer = TfidfVectorizer(max_features=5)
    train_vectors = vectorizer.fit_transform(df['script'])

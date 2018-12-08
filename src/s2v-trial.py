import gensim
import pandas as pd
import re





def make_script_set_of_tokens(str):
    '''
    this function creates a set of tokens from a string (in this case,
    from a movie script, which is a string)
    '''
    word_list = re.sub("[^\w]", " ", str).split()
    word_set = set(word_list)
    return word_set



if __name__ == '__main__':
    df = pd.read_csv('../data/lem-scripts')
    # nlp = spacy.load('en_core_web_lg')
    # s2v = Sense2VecComponent(df['script'][0])
    # nlp.add_pipe(s2v)

    scripts_list = df['script'].apply(make_script_set_of_tokens).tolist()

    model = gensim.models.Word2Vec(
    scripts_list,
    size=150,
    window=10,
    min_count=2,
    workers=-1,
    compute_loss=True)
    model.train(scripts_list, total_examples=len(scripts_list), epochs=10,
        compute_loss=True)

    # f = map(make_script_set_of_tokens, scripts_list)

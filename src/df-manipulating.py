import pandas as pd
import numpy as np

def find_len_of_script(str):
    return len(str)

def find_category_of_script(int):
    if int > 85:
        return 3
    elif int > 60 and int <= 85:
        return 2
    else:
        return 1



if __name__ == '__main__':
    sc_df = pd.read_csv('../data/scripts_df_II')
    sc_df['script_length'] = sc_df['script'].apply(find_len_of_script)
    sc_df.drop(['Unnamed: 0', 'index'], axis=1, inplace=True)
    df = sc_df.query('script_length > 10000')
    df['category'] = df['rating'].apply(find_category_of_script)
    df.to_csv('../data/scripts-rating-df')

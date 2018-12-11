import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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


if __name__ == '__main__':
    full_df = pd.read_csv('../data/scripts-rating-df')
    lem_df = pd.read_csv('../data/lem-scripts')
    lem_df['r>75'] = lem_df.apply(f, axis=1)

    # fig1, ax = plt.subplots()
    # ax.hist(full_df['rating'], bins=250)
    # ax.set_xlim(0,100)
    # ax.set_xlabel('Movie Ratings', fontsize=14)
    # ax.set_ylabel('Number of Movies', fontsize=14)
    # ax.set_title('Histogram of Movie Ratings', fontsize=14)
    # plt.savefig('../images/hist-of-movie-ratings')

    # fig1, ax = plt.subplots()
    # ax.hist(full_df['script_length'], bins=100)
    # ax.set_xlabel('Script Length (# of words)', fontsize=14)
    # ax.set_ylabel('Number of Movies', fontsize=14)
    # ax.set_title('Histogram of Movie Length (# of words)', fontsize=14)
    # plt.savefig('../images/hist-of-scrp-length')

    # fig1, ax = plt.subplots()
    # ax.scatter(full_df['rating'], full_df['script_length'], c='g')
    # ax.set_xlabel('Movie Rating', fontsize=14)
    # ax.set_xlim(0,105)
    # ax.set_ylabel('Words in Script (thousands)', fontsize=14)
    # ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x/1000))))
    # ax.set_title('Scatterplot of Ratings vs. # of Words', fontsize=14)
    # plt.savefig('../images/scatter-ratings-vs-words')

    # font = {'weight': 'bold', 'size': 16}
    # fig1, ax = plt.subplots()
    # fig1.subplots_adjust(bottom=0.2)
    # ax.bar(1, 558 , .2)
    # ax.bar(2, 501, .2)
    # ax.set_xlabel(" </= 75% = 'Bad' \n > 75% = 'Good'", fontsize=14)
    # ax.set_ylabel('Movie Count', fontsize=14)
    # ax.set_title(" Count of 'Good' vs 'Bad' Movies", fontsize=14)
    # ax.set_xticks([1, 2])
    # ax.set_xticklabels(("'Bad'", "'Good'"))
    # ax.text(1, 300, '558', font, horizontalalignment='center')
    # ax.text(2, 300, '501', font, horizontalalignment='center')
    # plt.savefig('../images/bar-good-vs-bad')

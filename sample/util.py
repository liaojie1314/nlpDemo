from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd


def get_word_frequency(movie_train):
    all_matrix = []
    all_words = []
    # 所有5个情感类别的总词频
    countVectorizer = CountVectorizer(min_df=0.0, max_df=1.0, ngram_range=(1, 2))
    countVectorizer.fit(movie_train.Phrase)
    print(len(countVectorizer.get_feature_names_out()))
    all_labels = ['negative', 'some-negative', 'neutral', 'some-positive', 'positive']
    neg_matrix = countVectorizer.transform(movie_train[movie_train.Sentiment == 0].Phrase)
    term_freq_df = pd.DataFrame(list(
        sorted([(word, neg_matrix.sum(axis=0)[0, idx]) for word, idx in countVectorizer.vocabulary_.items()],
               key=lambda x: x[1],
               reverse=True)), columns=['Terms', 'negative'])
    term_freq_df = term_freq_df.set_index('Terms')
    for i in range(1, 5):
        all_matrix.append(countVectorizer.transform(movie_train[movie_train.Sentiment == i].Phrase))
        all_words.append(all_matrix[i - 1].sum(axis=0))
        aa = pd.DataFrame(list(
            sorted([(word, all_words[i - 1][0, idx]) for word, idx in countVectorizer.vocabulary_.items()],
                   key=lambda x: x[1],
                   reverse=True)), columns=['Terms', all_labels[i]])

        term_freq_df = term_freq_df.join(aa.set_index('Terms'), how='left', lsuffix='_A')
    term_freq_df['total'] = term_freq_df['negative'] + term_freq_df['some-negative'] + term_freq_df['neutral'] + \
                            term_freq_df['some-positive'] + term_freq_df['positive']
    term_freq_df.sort_values(by='total', ascending=False).head(10)
    return term_freq_df

# def filter_worlds(movie_train):
#     phrases = movie_train[movie_train.Sentiment == 0]
#     words = []
#     for t in phrases.Phrase:
#         words.append(t)
#     neg_text = pd.Series(words).str.cat(sep=' ')
#     for t in phrases.Phrase[:300]:
#         if 'good' in t:
#             print(t)
#     # 即使文本包含“好”这样的词，也有可能是一种负面情绪
#     pos_phrases = movie_train[movie_train.Sentiment == 4]  # positive
#     pos_string = []
#     for t in pos_phrases.Phrase:
#         pos_string.append(t)
#     pos_text = pd.Series(pos_string).str.cat(sep=' ')
#     print(pos_text[:100])

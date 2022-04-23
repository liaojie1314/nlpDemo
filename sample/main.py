from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import train_test_split
from evaluation import *
from util import *

# 初始化数据
movie_train = pd.read_csv("../data/train.tsv", sep='\t')
movie_test = pd.read_csv("../data/test.tsv", sep='\t')

# movie_train.Sentiment.value_counts()
# movie_train.info()

# 删除不需要的列
movie_train_final = movie_train.drop(['PhraseId', 'SentenceId'], axis=1)
# print(movie_train_final.head())

movie_train_final['phrase_len'] = [len(t) for t in movie_train_final.Phrase]
# print(movie_train_final.head(4))

# fig, ax = plt.subplots(figsize=(5, 5))
# plt.boxplot(movie_train_final.phrase_len)
# plt.show()

# get_word_frequency(movie_train_final)

# 特征工程
phrase = np.array(movie_train_final['Phrase'])
sentiments = np.array(movie_train_final['Sentiment'])
# 创建训练和测试数据集
phrase_train, phrase_test, sentiments_train, sentiments_test = train_test_split(
    phrase,
    sentiments,
    test_size=0.2,
    random_state=4
)

# CountVectorizern Test
# cv1 = CountVectorizer()
# x_train_cv = cv1.fit_transform(["How are you", "Hi what's up", "What are you doing"])
# x_train_cv_df = pd.DataFrame(x_train_cv.toarray(), columns=list(cv1.get_feature_names_out()))
# print(x_train_cv_df)

# CountVectorizer参数设置
# 创建Bag-Of-Words
cv = CountVectorizer(stop_words='english', max_features=10000)
cv_train_features = cv.fit_transform(phrase_train)

# 创建TF-IDF
tv = TfidfVectorizer(min_df=0.0, max_df=1.0, ngram_range=(1, 2),
                     sublinear_tf=True)
tv_train_features = tv.fit_transform(phrase_train)
cv_test_features = cv.transform(phrase_test)
tv_test_features = tv.transform(phrase_test)
print('BOW model:> Train features shape:', cv_train_features.shape, ' Test features shape:', cv_test_features.shape)
print('TFIDF model:> Train features shape:', tv_train_features.shape, ' Test features shape:', tv_test_features.shape)

logisticRegression = LogisticRegression(penalty='l2', max_iter=1000, C=1)
sgd = SGDClassifier(loss='hinge')
# 基于CountVectorizer 上的逻辑回归模型
lr_bow_predictions = train_predict_model(classifier=logisticRegression,
                                         train_features=cv_train_features, train_labels=sentiments_train,
                                         test_features=cv_test_features, test_labels=sentiments_test)
display_model_performance_metrics(true_labels=sentiments_test, predicted_labels=lr_bow_predictions,
                                  classes=[0, 1, 2, 3, 4])

# 基于 TF-IDF 特征的逻辑回归模型
lr_tfidf_predictions = train_predict_model(classifier=logisticRegression,
                                           train_features=tv_train_features, train_labels=sentiments_train,
                                           test_features=tv_test_features, test_labels=sentiments_test)
display_model_performance_metrics(true_labels=sentiments_test, predicted_labels=lr_tfidf_predictions,
                                  classes=[0, 1, 2, 3, 4])

# 基于Countvectorizer的SGD模型
sgd_bow_predictions = train_predict_model(classifier=sgd,
                                          train_features=cv_train_features, train_labels=sentiments_train,
                                          test_features=cv_test_features, test_labels=sentiments_test)
display_model_performance_metrics(true_labels=sentiments_test, predicted_labels=sgd_bow_predictions,
                                  classes=[0, 1, 2, 3, 4])

# 基于TF-IDF的SGD模型
sgd_tfidf_predictions = train_predict_model(classifier=sgd,
                                            train_features=tv_train_features, train_labels=sentiments_train,
                                            test_features=tv_test_features, test_labels=sentiments_test)
display_model_performance_metrics(true_labels=sentiments_test, predicted_labels=sgd_tfidf_predictions,
                                  classes=[0, 1, 2, 3, 4])

# 基于TF-IDF的随机森林模型
rfc = RandomForestClassifier(n_jobs=-1)
rfc_tfidf_predictions = train_predict_model(classifier=rfc,
                                            train_features=tv_train_features, train_labels=sentiments_train,
                                            test_features=tv_test_features, test_labels=sentiments_test)
display_model_performance_metrics(true_labels=sentiments_test, predicted_labels=rfc_tfidf_predictions,
                                  classes=[0, 1, 2, 3, 4])

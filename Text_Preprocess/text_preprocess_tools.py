from warnings import filterwarnings

import pandas as pd
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from textblob import Word

filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

data_set = pd.read_excel(r'C:\Users\pc\SentimentAnalysis\Data_Set\amazon.xlsx')

def text_preprocessing(dataframe):
    dataframe['Review'] = dataframe['Review'].str.lower()
    dataframe['Review'] = dataframe['Review'].str.replace('[^\w\s]', '', regex=True)
    dataframe['Review'] = dataframe['Review'].str.replace('\d', '', regex=True)

    stop_words_remove = set(stopwords.words('english'))
    dataframe['Review'] = dataframe['Review'].apply(lambda x: ' '.join(word for word in str(x).split() if word not in stop_words_remove))
    word_frequencies = pd.Series(' '.join(dataframe['Review']).split()).value_counts()
    low_words = word_frequencies[word_frequencies <= 1000]

    dataframe['Review'] = dataframe['Review'].apply(lambda x: ' '.join(word for word in str(x).split() if word not in low_words))
    dataframe['Review'] = dataframe['Review'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

text_preprocessing(data_set)


def sentiment_analysis(dataframe):
    sentiment_score_calculator = SentimentIntensityAnalyzer()
    dataframe['Sentiment_Scores'] = dataframe['Review'].apply(
        lambda x: 1 if sentiment_score_calculator.polarity_scores(x)['compound'] > 0 else 0)

sentiment_analysis(data_set)

def train_test_sep(dataframe):
    dataframe = dataframe.drop('Title', axis=1)
    y = dataframe['Sentiment_Scores']
    X = dataframe['Review']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return dataframe, X_train, X_test, y_train, y_test

data_set, X_train, X_test, y_train, y_test = train_test_sep(data_set)

def world_to_vec(Xtrain,Xtest):
    tf_idf_word_vectorizer = TfidfVectorizer()
    X_train_tf_idf_word = tf_idf_word_vectorizer.fit_transform(Xtrain)
    X_test_tf_idf_word = tf_idf_word_vectorizer.fit_transform(Xtest)

    return X_train_tf_idf_word, X_test_tf_idf_word

X_train_tf_idf_word, X_test_tf_idf_word = world_to_vec(X_train, X_test)





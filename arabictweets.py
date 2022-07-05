import numpy as np
import pandas as pd
import string
import re
import nltk
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
from textblob import TextBlob


columnNames =['Type', 'Tweets']
negativeTweets = pd.read_csv(r'C:\Users\manar\Downloads\Compressed\Arabic tweets\train_Arabic_tweets_negative_20190413.tsv',sep='\t', names = columnNames)
positiveTweets = pd.read_csv(r'C:\Users\manar\Downloads\Compressed\Arabic tweets\train_Arabic_tweets_positive_20190413.tsv',sep='\t', names = columnNames)
pd.set_option('display.max_colwidth', 100)



def remove_punctuations(tweet):
    no_panctuation ="".join([c for c in tweet if c not in string.punctuation])
    return no_panctuation


def preprocess(tweet):

    tweet = re.sub(r'[0-9]+', '', tweet)
    tweet = re.sub(r'\s*[A-Za-z]+\b', '', tweet)
    tweet = tweet.rstrip()
    tweet = tweet.replace("_", " ")
    tweet = re.sub(r'\s+', ' ', tweet)
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    tweet = remove_punctuations(tweet)
    tweet = ' '.join(dict.fromkeys(tweet.split()))
    return tweet


def listToString(tweet):
    str1 = ' '.join(map(str, tweet))
    return str1

positiveTweets["Tweets"] = positiveTweets['Tweets'].apply(lambda x: preprocess(x))
negativeTweets["Tweets"] = negativeTweets['Tweets'].apply(lambda x: preprocess(x))

#print(negativeTweets['Tweets'].head())
#print(positiveTweets['Tweets'].head())

tokenizer = RegexpTokenizer(r'\w+')
positiveTweets["Tweets"] = positiveTweets["Tweets"].apply(tokenizer.tokenize)
negativeTweets["Tweets"] = negativeTweets['Tweets'].apply(tokenizer.tokenize)

#print(negativeTweets['Tweets'].head())
#print(positiveTweets['Tweets'].head())

stopwords_list = set(stopwords.words('arabic'))
positiveTweets["Tweets"]=positiveTweets["Tweets"].apply(lambda x: [item for item in x if item not in stopwords_list])
negativeTweets["Tweets"] = negativeTweets['Tweets'].apply(lambda x: [item for item in x if item not in stopwords_list])
positiveTweets["Tweets"] = positiveTweets['Tweets'].apply(lambda x: listToString(x))

#print(positiveTweets['Tweets'])


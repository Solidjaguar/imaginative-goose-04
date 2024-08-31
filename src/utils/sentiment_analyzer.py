import requests
from textblob import TextBlob
from newsapi import NewsApiClient
import tweepy
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    def __init__(self, news_api_key, twitter_api_key, twitter_api_secret, twitter_access_token, twitter_access_token_secret):
        self.newsapi = NewsApiClient(api_key=news_api_key)
        auth = tweepy.OAuthHandler(twitter_api_key, twitter_api_secret)
        auth.set_access_token(twitter_access_token, twitter_access_token_secret)
        self.twitter_api = tweepy.API(auth)

    def get_news_sentiment(self, query, from_param, to):
        try:
            articles = self.newsapi.get_everything(q=query,
                                                   from_param=from_param,
                                                   to=to,
                                                   language='en',
                                                   sort_by='relevancy')
            
            sentiments = []
            for article in articles['articles']:
                blob = TextBlob(article['title'] + ' ' + article['description'])
                sentiments.append(blob.sentiment.polarity)
            
            return pd.Series(sentiments).mean() if sentiments else 0
        except Exception as e:
            logger.error(f"Error fetching news sentiment: {str(e)}")
            return 0

    def get_twitter_sentiment(self, query, count=100):
        try:
            tweets = self.twitter_api.search_tweets(q=query, count=count, lang='en')
            sentiments = []
            for tweet in tweets:
                blob = TextBlob(tweet.text)
                sentiments.append(blob.sentiment.polarity)
            
            return pd.Series(sentiments).mean() if sentiments else 0
        except Exception as e:
            logger.error(f"Error fetching Twitter sentiment: {str(e)}")
            return 0

    def get_combined_sentiment(self, query, from_param, to):
        news_sentiment = self.get_news_sentiment(query, from_param, to)
        twitter_sentiment = self.get_twitter_sentiment(query)
        return (news_sentiment + twitter_sentiment) / 2

# You can add more sentiment analysis functions here as needed
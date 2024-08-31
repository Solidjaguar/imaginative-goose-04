from newsapi import NewsApiClient
import pandas as pd
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class NewsResearcher:
    def __init__(self, news_api_key):
        self.newsapi = NewsApiClient(api_key=news_api_key)

    def get_news_around_date(self, date, days_before=3, days_after=3, query='gold'):
        start_date = (date - timedelta(days=days_before)).strftime('%Y-%m-%d')
        end_date = (date + timedelta(days=days_after)).strftime('%Y-%m-%d')

        try:
            articles = self.newsapi.get_everything(q=query,
                                                   from_param=start_date,
                                                   to=end_date,
                                                   language='en',
                                                   sort_by='relevancy')
            
            news_data = []
            for article in articles['articles']:
                news_data.append({
                    'date': article['publishedAt'],
                    'title': article['title'],
                    'description': article['description'],
                    'url': article['url']
                })
            
            return pd.DataFrame(news_data)
        except Exception as e:
            logger.error(f"Error fetching news: {str(e)}")
            return pd.DataFrame()

    def analyze_news_for_trades(self, trades_df, query='gold'):
        all_news = []
        for _, trade in trades_df.iterrows():
            date = pd.to_datetime(trade['date'])
            news = self.get_news_around_date(date, query=query)
            news['trade_date'] = date
            news['trade_return'] = trade['trade_return']
            all_news.append(news)
        
        return pd.concat(all_news, ignore_index=True)

    def summarize_news(self, news_df):
        # Here you could implement more sophisticated news summarization
        # For now, we'll just return the top 5 most relevant articles
        return news_df.sort_values('relevancy', ascending=False).head()

# You can add more news research methods here as needed
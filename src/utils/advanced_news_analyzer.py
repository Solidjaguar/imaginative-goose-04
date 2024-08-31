import pandas as pd
import numpy as np
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import spacy
import logging

logger = logging.getLogger(__name__)

class AdvancedNewsAnalyzer:
    def __init__(self):
        self.sentiment_analyzer = pipeline("sentiment-analysis")
        self.nlp = spacy.load("en_core_web_sm")
        self.tfidf = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
        self.lda = LatentDirichletAllocation(n_components=5, random_state=42)

    def analyze_sentiment(self, texts):
        sentiments = self.sentiment_analyzer(texts)
        return [s['label'] for s in sentiments], [s['score'] for s in sentiments]

    def extract_entities(self, texts):
        entities = []
        for text in texts:
            doc = self.nlp(text)
            entities.append([(ent.text, ent.label_) for ent in doc.ents])
        return entities

    def extract_topics(self, texts):
        tfidf_matrix = self.tfidf.fit_transform(texts)
        self.lda.fit(tfidf_matrix)
        
        topic_words = []
        for topic_idx, topic in enumerate(self.lda.components_):
            top_words = [self.tfidf.get_feature_names()[i] for i in topic.argsort()[:-10 - 1:-1]]
            topic_words.append(top_words)
        
        return topic_words

    def analyze_news(self, news_df):
        texts = news_df['title'] + ' ' + news_df['description']
        
        sentiments, sentiment_scores = self.analyze_sentiment(texts)
        entities = self.extract_entities(texts)
        topics = self.extract_topics(texts)
        
        news_df['sentiment'] = sentiments
        news_df['sentiment_score'] = sentiment_scores
        news_df['entities'] = entities
        news_df['topics'] = [topics] * len(news_df)  # Assigning the same topics to all rows
        
        return news_df

    def summarize_analysis(self, analyzed_df):
        summary = {
            'overall_sentiment': analyzed_df['sentiment'].mode().iloc[0],
            'avg_sentiment_score': analyzed_df['sentiment_score'].mean(),
            'top_entities': self.get_top_entities(analyzed_df['entities']),
            'topics': analyzed_df['topics'].iloc[0]  # Topics are the same for all rows
        }
        return summary

    def get_top_entities(self, entities_list, top_n=5):
        all_entities = [ent for sublist in entities_list for ent in sublist]
        entity_counts = pd.Series(all_entities).value_counts()
        return entity_counts.head(top_n).to_dict()

# You can add more advanced NLP methods here as needed
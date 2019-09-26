import requests
import json
from topic_identify.bean import Article
from elasticsearch import Elasticsearch
import time
import datetime


class FeedsContent:
    def __init__(self):
        self.es = Elasticsearch('http://10.101.93.234:9810/')
        self.url = 'http://10.101.93.234:9810/_search'
        self.payload1 = {
            "query": {
                "bool": {
                    "must": [
                        {
                            "multi_match": {
                                "query": "{1}",
                                "fields": [
                                    "title",
                                    "content"
                                ]
                            }
                        }
                    ],
                    "filter": {
                        "range": {
                            "postTime": {
                                "gt": 1569340800000
                            }
                        }
                    }
                }
            },
            "sort": [
                {
                    "postTime": {
                        "order": "desc"
                    }
                }
            ],
            "size": 2
        }
        self.payload2 = {
            'query': {
                'bool': {
                    'must': [
                        {'multi_match': {
                            'query': '{0}',
                            'fields': [
                                'title',
                                'content'
                            ]
                        }},
                        {'multi_match': {
                            'query': '{1}',
                            'fields': [
                                'title',
                                'content'
                            ]
                        }}
                    ]
                }
            }
        }
        self.payload3 = {
            'query': {
                'bool': {
                    'must': [
                        {'multi_match': {
                            'query': '{0}',
                            'fields': [
                                'title',
                                'content'
                            ]
                        }},
                        {'multi_match': {
                            'query': '{1}',
                            'fields': [
                                'title',
                                'content'
                            ]
                        }},
                        {'multi_match': {
                            'query': '{2}',
                            'fields': [
                                'title',
                                'content'
                            ]
                        }}
                    ]
                }
            }
        }

    def get_articles(self, keywords, size, timestamp):
        response = requests.post(self.url, json=self.get_query(keywords, size, timestamp))
        json_obj = json.loads(response.text)
        articles_json = json_obj.get('hits').get('hits')

        articles = []
        for article_json in articles_json:
            article = Article()

            article.id = article_json.get('_source').get('articleId')
            article.title = article_json.get('_source').get('title')
            article.content = article_json.get('_source').get('content')
            articles.append(article)

        return articles

    def get_articles_from_es(self, keywords):

        query = self.get_query(['习近平'])
        yestoday = datetime.date.today() - datetime.timedelta(days=1)
        timestamp = time.mktime(yestoday.timetuple()) * 1000

        feeds_content = FeedsContent()
        articles = feeds_content.get_articles(['习近平213123'], 2, timestamp)
        # for article in articles:
        #     # print(article.__dict__)
        return articles


    def get_query(self, keywords, size, timestamp):
        keywords_len = len(keywords)
        if keywords_len == 3:
            self.payload3['query']['bool']['must'][0]['multi_match']['query'] = keywords[0]
            self.payload3['query']['bool']['must'][1]['multi_match']['query'] = keywords[1]
            self.payload3['query']['bool']['must'][2]['multi_match']['query'] = keywords[2]
            return self.payload3
        elif keywords_len == 2:
            self.payload2['query']['bool']['must'][0]['multi_match']['query'] = keywords[0]
            self.payload2['query']['bool']['must'][1]['multi_match']['query'] = keywords[1]
            return self.payload2
        else:
            self.payload1['query']['bool']['must'][0]['multi_match']['query'] = keywords[0]
            self.payload1['size'] = size
            self.payload1['query']['bool']['filter']['range']['postTime']['gt'] = timestamp
            return self.payload1


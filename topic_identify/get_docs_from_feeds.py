import requests
import json


class FeedsContent:
    def __init__(self):
        self.url = 'http://10.101.93.234:9810/_search'
        self.payload1 = {
            'query': {
                'multi_match': {
                    'query': '{}',
                    'fields': [
                        'title',
                        'content'
                    ]
                }
            }
        }
        self.payload2 = {
            'query': {
                'bool': {
                    'should': [
                        {
                            'multi_match': {
                                'query': '{}',
                                'fields': [
                                    'title',
                                    'content'
                                ]
                            },
                            'multi_match': {
                                'query': '{}',
                                'fields': [
                                    'title',
                                    'content'
                                ]
                            }
                        }
                    ]
                }
            }
        }

    def getDocs(self, keyword):
        self.payload1['query']['multi_match']['query'] = keyword
        response = requests.post(self.url, json=self.payload1)
        json_obj = json.loads(response.text)
        articles = json_obj['hits']['hits']

        result = []
        for article in articles:
            article_dict = {}
            article_dict['articleId'] = article['_source']['articleId']
            article_dict['title'] = article['_source']['title']
            article_dict['content'] = article['_source']['content']
            result.append(article_dict)

        return result


if __name__ == '__main__':
    feeds_content = FeedsContent()
    feeds_content.getDocs('习近平')

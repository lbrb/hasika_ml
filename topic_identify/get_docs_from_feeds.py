import requests

url = 'http://10.101.93.234:9810/'
payload = {
  "query": {
    "multi_match": {
      "query": "新城",
      "fields": [
        "title",
        "content"
      ]
    }
  }
}

response = requests.post(url, data=payload)
print(response)
artices = response['hits']['_source']



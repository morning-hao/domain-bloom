import requests

url = 'http://localhost:5000/'

question = "怎么提高睡眠质量"

response = requests.post(url, json={'question': question})

print(response.json())

import requests

API_KEY = "d78a3ba178fa4358a627bf74509301af"
url = f"https://newsapi.org/v2/top-headlines?country=in&language=en&pageSize=5&apiKey={API_KEY}"

response = requests.get(url)
print("Status Code:", response.status_code)
print("Response JSON:", response.json())

import requests
import pickle

# Load your trained model and vectorizer
model = pickle.load(open("news_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# Your NewsAPI key
API_KEY = "d78a3ba178fa4358a627bf74509301af"

def fetch_headlines():
    url = f"https://newsapi.org/v2/top-headlines?language=en&pageSize=5&apiKey={API_KEY}"
    response = requests.get(url)
    print("Status code:", response.status_code)  # Debugging
    data = response.json()
    print("Raw API Response:", data)  # Debugging
    if "articles" not in data:
        print("⚠️ Error from NewsAPI:", data)
        return []
    articles = data["articles"]
    return [article["title"] for article in articles]


def classify_headlines(headlines):
    if not headlines:
        print("⚠️ No headlines fetched.")
        return
    for i, headline in enumerate(headlines, 1):
        vec = vectorizer.transform([headline])
        pred = model.predict(vec)[0]
        label = "REAL NEWS ✅" if pred == 1 else "FAKE NEWS ❌"
        print(f"{i}. {headline}\n   → {label}\n")

if __name__ == "__main__":
    headlines = fetch_headlines()
    classify_headlines(headlines)

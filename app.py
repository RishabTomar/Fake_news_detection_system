from flask import Flask, render_template , request
import pickle
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import re
import translator


app = Flask(__name__)

# Load model + vectorizer
model = pickle.load(open("news_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# Cleaning function (same as training)
def clean(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

# Scraper
def fetch_headlines():
    url = "https://news.google.com/rss?hl=en-IN&gl=IN&ceid=IN:en"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "lxml-xml")
    headlines = [item.title.text for item in soup.find_all("item")][:5]
    return headlines

# Classification
def classify_headlines(headlines):
    results = []
    for headline in headlines:
        cleaned = clean(headline)
        vec = vectorizer.transform([cleaned])

        pred = model.predict(vec)[0]
        proba = model.predict_proba(vec)[0]

        print("\n==============================")
        print("RAW HEADLINE:", headline)
        print("CLEANED:", cleaned)
        print("PRED:", pred)
        print("PROBA:", proba)
        print("==============================\n")

        label = "REAL NEWS ✅" if pred == 1 else "FAKE NEWS ❌"
        results.append((headline, label))

    return results

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['news']

    translated_text = translator.translate_to_english(text)
    cleaned = clean(translated_text)

    vec = vectorizer.transform([cleaned])
    prediction = model.predict(vec)[0]

    result = "REAL NEWS ✅" if prediction == 1 else "FAKE NEWS ❌"

    return render_template("index.html",
                           prediction=result,
                           original=text,
                           translated=translated_text)

@app.route('/live-news')
def live_news():
    headlines = fetch_headlines()
    classified = classify_headlines(headlines)
    last_updated = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

    return render_template("index.html",
                           live_news=classified,
                           last_updated=last_updated)

if __name__ == "__main__":
    app.run(debug=True)

import pandas as pd
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.utils import resample

# Load data
fake = pd.read_csv("Fake.csv")
real = pd.read_csv("True.csv")

fake['label'] = 0
real['label'] = 1

# 🔥 Use ONLY balanced data — do NOT overwrite later
df = pd.concat([fake, real])[['text', 'label']]

print("Before balancing:", df['label'].value_counts())

fake_df = df[df['label'] == 0]
real_df = df[df['label'] == 1]

# Balance dataset
if len(fake_df) > len(real_df):
    fake_df = resample(fake_df, replace=False, n_samples=len(real_df), random_state=42)
else:
    real_df = resample(real_df, replace=False, n_samples=len(fake_df), random_state=42)

df = pd.concat([fake_df, real_df]).sample(frac=1, random_state=42)

print("AFTER BALANCING:", df['label'].value_counts())

# Clean
def clean(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

df['text'] = df['text'].apply(clean)

# TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X = vectorizer.fit_transform(df['text'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(class_weight="balanced", max_iter=3000)
model.fit(X_train, y_train)

pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))

# Save files
pickle.dump(model, open("news_model.pkl", "wb"))
pickle.dump(vectorizer, open("tfidf_vectorizer.pkl", "wb"))

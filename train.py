import json
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

nltk.download("punkt")
nltk.download("stopwords")

stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

with open("intents.json") as file:
    data = json.load(file)

corpus = []
tags = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        # Tokenize
        tokens = nltk.word_tokenize(pattern.lower())

        # Remove stop words
        filtered = [word for word in tokens if word.isalnum() and word not in stop_words]

        # stemming
        stemmed = [stemmer.stem(word) for word in filtered]

        processed_text = " ".join(stemmed)
        corpus.append(processed_text)
        tags.append(intent["tag"])

# Bag of Words
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
y = tags

# Train the model
model = MultinomialNB()
model.fit(X, y)

# Save the model and vectorizer
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
with open("tags.pkl", "wb") as f:
    pickle.dump(tags, f)

print("✅ Model trained and saved")

import json
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import sqlite3

nltk.download("punkt")
nltk.download("stopwords")

stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()
DB_PATH = "database.db"

with open("intents.json") as file:
    data = json.load(file)

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
tags = pickle.load(open("tags.pkl", "rb"))

def clean_text(text):
    tokens = nltk.word_tokenize(text.lower())
    filtered = [word for word in tokens if word.isalnum() and word not in stop_words]
    stemmed = [stemmer.stem(word) for word in filtered]
    return " ".join(stemmed)

def get_response(text):
    cleaned = clean_text(text)
    X = vectorizer.transform([cleaned])
    probs = model.predict_proba(X)[0]
    max_prob = max(probs)
    tag_index = probs.argmax()
    print("Answer Prob: ", max_prob)

    if max_prob < 0.07:
        return "unknown", "Sorry, ", max_prob

    tag = model.classes_[tag_index]
    for intent in data["intents"]:
        if intent["tag"] == tag:
            return tag, intent["responses"][0], max_prob

    return "unknown", "Sorry, I do not understand. I will forward this to a counselor.", max_prob

def retrain_model():
    corpus = []
    tags_list = []

    # Existing intents
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            corpus.append(pattern)
            tags_list.append(intent["tag"])

    # New answered unknown questions
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT question, answer FROM unknown_questions WHERE answer IS NOT NULL")
    rows = c.fetchall()
    conn.close()

    for question, answer in rows:
        corpus.append(question)
        tags_list.append(answer)

    # Apply preprocessing (tokenize, remove stopwords, stem)
    processed_corpus = []
    for text in corpus:
        tokens = nltk.word_tokenize(text.lower())
        filtered = [word for word in tokens if word.isalnum() and word not in stop_words]
        stemmed = [stemmer.stem(word) for word in filtered]
        processed_corpus.append(" ".join(stemmed))

    # Train new model
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(processed_corpus)
    y = tags_list

    model = MultinomialNB()
    model.fit(X, y)

    # Save updated model and vectorizer
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    with open("tags.pkl", "wb") as f:
        pickle.dump(list(set(y)), f)

import re
import wikipedia
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import numpy as np

def tokenize(text):
    return re.findall(r'\w+', text.lower())

def get_wiki_content(topic):
    try:
        return wikipedia.page(topic, auto_suggest=True).content
    except wikipedia.exceptions.DisambiguationError as e:
        return wikipedia.page(e.options[0]).content
    except Exception as e:
        print(f"Error fetching {topic}: {e}")
        return ""

def average_word2vec(tokens, model, size):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(size)

def main():
    topics = [
        "Galaxy",
        "Natural language processing",
        "Nebula",
        "Logistic regression",
        "Artificial intelligence"
    ]

    tokenized_docs = []
    labels = []

    for topic in topics:
        print(f"Fetching: {topic}")
        content = get_wiki_content(topic)
        tokens = tokenize(content)
        tokenized_docs.append(tokens)
        labels.append(topic)

    # Train Word2Vec
    vector_size = 100
    model = Word2Vec(sentences=tokenized_docs, vector_size=vector_size, window=5, min_count=2, workers=2, epochs=50)

    # Convert docs to vectors
    doc_vectors = np.array([average_word2vec(doc, model, vector_size) for doc in tokenized_docs])

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(labels)

    # Train Logistic Regression classifier
    clf = LogisticRegression(max_iter=200)
    clf.fit(doc_vectors, y)
    y_pred = clf.predict(doc_vectors)

    # Show results
    print("Labels:", labels)
    print("Predictions:", le.inverse_transform(y_pred))

if __name__ == "__main__":
    main()

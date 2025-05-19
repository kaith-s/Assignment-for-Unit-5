import math
import re
import wikipedia
from collections import Counter

# Set Wikipedia to English
wikipedia.set_lang("en")

# utils

def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

def compute_tf(tokens, vocab):
    total = len(tokens) or 1
    counts = Counter(tokens)
    return {term: counts[term] / total for term in vocab}

def cosine_similarity(vec1, vec2, vocab):
    dot = sum(vec1[term] * vec2[term] for term in vocab)
    len1 = math.sqrt(sum(vec1[term]**2 for term in vocab))
    len2 = math.sqrt(sum(vec2[term]**2 for term in vocab))
    return dot / (len1 * len2) if len1 and len2 else 0.0

def get_article(topic):
    try:
        return wikipedia.page(topic).content
    except:
        return ""

# main code

def main():
    topics = ["Black hole", "Neural network", "Climate change"]
    
    # Get and tokenize documents
    docs = []
    for topic in topics:
        content = get_article(topic)
        tokens = tokenize(content)
        docs.append(tokens)
        print(f"{topic} → {len(tokens)} tokens")

    # create vocab
    vocab = sorted(set(word for doc in docs for word in doc))

    # compute TF vectors
    tf_vectors = [compute_tf(doc, vocab) for doc in docs]

    # compare first doc with others
    print("\nCosine Similarity:")
    for i in range(1, len(tf_vectors)):
        score = cosine_similarity(tf_vectors[0], tf_vectors[i], vocab)
        print(f"{topics[0]} vs {topics[i]}: {round(score, 4)}")

if __name__ == "__main__":
    main()

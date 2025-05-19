import re
import wikipedia
import math
from collections import Counter

# Set language to English
wikipedia.set_lang("en")

# Utils

def tokenize(text):
    # Lowercase and extract alphanumeric words
    return re.findall(r'\b\w+\b', text.lower())

def compute_tf(doc_tokens, vocab):
    counts = Counter(doc_tokens)
    total_terms = len(doc_tokens) or 1
    return {term: counts[term] / total_terms for term in vocab}

def compute_idf(all_docs, vocab):
    num_docs = len(all_docs)
    return {
        term: math.log(num_docs / (1 + sum(1 for doc in all_docs if term in doc)))
        for term in vocab
    }

def compute_tfidf(tf_vec, idf_vec, vocab):
    return {term: tf_vec[term] * idf_vec[term] for term in vocab}

def get_article(topic):
    try:
        return wikipedia.page(topic, auto_suggest=True).content
    except wikipedia.DisambiguationError as e:
        try:
            return wikipedia.page(e.options[0]).content
        except:
            return ""
    except:
        return ""

# Main code

def main():
    topics = [
        "Black hole",
        "Quantum mechanics",
        "Neural network",
        "Data mining",
        "Climate change"
    ]
    
    docs = []
    tokenized = {}
    
    for topic in topics:
        print(f"Processing: {topic}")
        content = get_article(topic)
        tokens = tokenize(content)
        docs.append(tokens)
        tokenized[topic] = tokens
        print(f"→ {len(tokens)} tokens collected.")

    # Create vocabulary
    vocab = sorted(set(term for doc in docs for term in doc))

    # Raw frequency matrix
    raw_freq = {topic: Counter(tokens) for topic, tokens in tokenized.items()}

    # Compute TF-IDF
    idf = compute_idf(docs, vocab)
    tfidf = {
        topic: compute_tfidf(compute_tf(tokens, vocab), idf, vocab)
        for topic, tokens in tokenized.items()
    }

    # Display samples
    def show(matrix, label):
        print(f"\n--- {label} ---")
        for topic in topics:
            top_terms = sorted(matrix[topic].items(), key=lambda x: -x[1])[:5]
            print(f"\n{topic}:")
            print({term: round(score, 4) for term, score in top_terms})
    
    show(raw_freq, "Raw Frequency (Top Terms)")
    show(tfidf, "TF-IDF (Top Terms)")

if __name__ == "__main__":
    main()

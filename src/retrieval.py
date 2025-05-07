from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

class Retriever:
    def __init__(self, documents):
        self.documents = documents
        self.vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        self.tfidf_matrix = self.vectorizer.fit_transform(documents)
    
    def preprocess_query(self, query):
        # Remove 'Q:' prefix and clean up
        query = re.sub(r'^\s*Q:\s*', '', query, flags=re.IGNORECASE)
        return query.strip()
    
    def retrieve(self, query, k=5, threshold=0.1):
        query = self.preprocess_query(query)
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        # Filter by threshold
        results = [(self.documents[i], similarities[i]) for i in top_k_indices if similarities[i] >= threshold]
        return results if results else [(self.documents[top_k_indices[0]], similarities[top_k_indices[0]])]
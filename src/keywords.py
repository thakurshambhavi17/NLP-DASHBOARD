from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

def extract_keywords(text, top_n=10):
    """Extract keywords from text using TF-IDF."""
    if not text or len(text.strip()) == 0:
        return []
        
    # We will treat each sentence as a document to find within-text frequency importance
    from src.preprocessing import split_sentences
    sentences = split_sentences(text)
    
    if len(sentences) < 2:
        # If it's too short, fake a few documents or use words
        sentences = [text]
        
    try:
        # Using bigrams and unigrams
        vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform(sentences)
        
        # Sum tfidf scores across all sentences
        sum_scores = tfidf_matrix.sum(axis=0)
        
        words = vectorizer.get_feature_names_out()
        scores = [(word, sum_scores[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
        
        # Sort by score descending
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
        
        return [word for word, score in sorted_scores[:top_n]]
    except Exception as e:
        return []


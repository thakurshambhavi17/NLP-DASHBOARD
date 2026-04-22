from src.preprocessing import split_sentences

def analyze_behavior(text):
    """Extract basic behavioral patterns like sentence length."""
    sentences = split_sentences(text)
    
    if not sentences:
        return {
            "total_sentences": 0,
            "avg_words_per_sentence": 0,
            "total_words": 0
        }
        
    words_per_sentence = [len(s.split()) for s in sentences]
    total_words = sum(words_per_sentence)
    avg_length = total_words / len(sentences)
    
    # Simple repetition checking (words that appear more than 5 times)
    word_counts = {}
    for word in text.lower().split():
        # filter out very short words
        if len(word) > 3:
            word_counts[word] = word_counts.get(word, 0) + 1
            
    repetitions = {w: c for w, c in word_counts.items() if c >= 3}
    
    return {
        "total_sentences": len(sentences),
        "avg_words_per_sentence": round(avg_length, 2),
        "total_words": total_words,
        "repetitions": repetitions
    }

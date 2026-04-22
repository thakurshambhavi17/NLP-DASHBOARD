def detect_personality(sentiment_result, behavior_result, keywords):
    """
    Hybrid logic combining sentiment, sentence stats, and word patterns 
    to output personality traits.
    """
    traits = []
    
    # 1. Emotional vs Analytical
    avg_len = behavior_result.get('avg_words_per_sentence', 0)
    sentiment_label = sentiment_result.get('label', 'NEUTRAL')
    sentiment_score = sentiment_result.get('score', 0)
    
    # Very short sentences with high sentiment -> highly emotional
    if avg_len < 10 and sentiment_score > 0.8:
        traits.append("Expressive / Emotional")
    elif avg_len > 18:
        traits.append("Analytical / Methodical")
    else:
        traits.append("Balanced Expression")
        
    # 2. Formal vs Casual
    formal_words = {'furthermore', 'moreover', 'therefore', 'consequently', 'regarding', 'ensure'}
    casual_words = {'like', 'you know', 'crazy', 'super', 'gonna', 'wanna'}
    
    text_words = set([k.lower() for k in keywords])
    
    if len(formal_words.intersection(text_words)) > 0:
        traits.append("Formal / Professional")
    elif len(casual_words.intersection(text_words)) > 1:
        traits.append("Casual / Conversational")
        
    # 3. Sentiment-based
    if sentiment_label == 'POSITIVE' and sentiment_score > 0.9:
        traits.append("Highly Optimistic")
    elif sentiment_label == 'NEGATIVE' and sentiment_score > 0.9:
        traits.append("Critical")
        
    return traits

from transformers import pipeline
from src.preprocessing import split_sentences

def load_sentiment_analyzer():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def analyze_sentiment(pipeline_obj, text):
    """Analyze the sentiment of the overall text."""
    # Truncate to first 512 tokens to avoid limits
    short_text = " ".join(text.split()[:400])
    try:
        result = pipeline_obj(short_text)
        return result[0]
    except Exception as e:
        return {"label": "NEUTRAL", "score": 0.0}

def track_sentiment_over_time(pipeline_obj, text):
    """Return sentiment scores for each sentence to visualize a timeline."""
    sentences = split_sentences(text)
    scores = []
    
    for s in sentences:
        if len(s.split()) < 3:
            continue
        try:
            res = pipeline_obj(s[:512])[0]
            # Convert NEGATIVE to negative score
            score = res['score'] if res['label'] == 'POSITIVE' else -res['score']
            scores.append(score)
        except:
            scores.append(0.0)
            
    return scores

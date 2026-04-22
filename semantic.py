from sentence_transformers import SentenceTransformer, util
import torch
from src.preprocessing import split_sentences

def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def semantic_search(model, text, query_sentences, top_k=3):
    """Finds sentences in the text that are most semantically similar to queries."""
    text_sentences = split_sentences(text)
    if not text_sentences:
        return {}
        
    text_embeddings = model.encode(text_sentences, convert_to_tensor=True)
    results = {}
    
    for query in query_sentences:
        query_embedding = model.encode(query, convert_to_tensor=True)
        cos_scores = util.cos_sim(query_embedding, text_embeddings)[0]
        
        # Get top-k
        top_results = torch.topk(cos_scores, k=min(top_k, len(text_sentences)))
        
        matches = []
        for score, idx in zip(top_results[0], top_results[1]):
            matches.append({
                "sentence": text_sentences[idx],
                "score": score.item()
            })
        results[query] = matches
        
    return results

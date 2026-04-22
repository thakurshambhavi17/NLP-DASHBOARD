import re

def clean_text(text):
    """Removes excess whitespace and basic noise."""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def split_sentences(text):
    """A simple regex-based sentence splitter if spacy isn't used here."""
    # Splitting on '.', '!', '?' followed by a space
    sentences = re.split(r'(?<=[.!?]) +', text)
    return [s.strip() for s in sentences if s.strip()]

def chunk_text(text, max_words=300):
    """Splits text into larger chunks (for models with token limits)."""
    words = clean_text(text).split(' ')
    chunks = []
    current_chunk = []
    
    for word in words:
        current_chunk.append(word)
        if len(current_chunk) >= max_words:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            
    if current_chunk:
        chunks.append(' '.join(current_chunk))
        
    return chunks

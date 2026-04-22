import spacy

def load_ner_model():
    return spacy.load("en_core_web_sm")

def extract_entities(nlp, text):
    """Extract Named Entities and return counts by type."""
    doc = nlp(text)
    entities = {}
    
    for ent in doc.ents:
        if ent.label_ not in entities:
            entities[ent.label_] = set()
        entities[ent.label_].add(ent.text)
        
    # Convert sets to sorted lists for JSON serialization
    return {label: sorted(list(names)) for label, names in entities.items()}

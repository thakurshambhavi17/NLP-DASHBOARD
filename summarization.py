from transformers import pipeline

def load_summarizer():
    # Cache the model in app.py, but this returns the pipeline
    return pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_text(pipeline_obj, text):
    """Summarize the text using the provided HF pipeline."""
    if len(text.split()) < 30:
        return "Text is too short to summarize."
        
    try:
        # Max length should be less than the length of the input
        input_length = len(text.split())
        max_len = min(130, max(50, int(input_length * 0.6)))
        min_len = min(30, max_len - 10)
        
        result = pipeline_obj(text, max_length=max_len, min_length=min_len, do_sample=False)
        return result[0]['summary_text']
    except Exception as e:
        return f"Error during summarization: {str(e)}"

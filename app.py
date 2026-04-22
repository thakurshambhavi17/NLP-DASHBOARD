import sys
import os
# Ensure Streamlit Cloud can find local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import torch

from src.preprocessing import chunk_text
from src.summarization import load_summarizer, summarize_text
from src.keywords import extract_keywords
from src.sentiment import load_sentiment_analyzer, analyze_sentiment, track_sentiment_over_time
from src.ner import load_ner_model, extract_entities
from src.semantic import load_embedding_model, semantic_search
from src.behavior import analyze_behavior
from src.personality import detect_personality

st.set_page_config(page_title="NLP Insights Dashboard", page_icon="🌌", layout="wide")

st.markdown("""
<style>
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
        font-family: 'Inter', sans-serif;
    }
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #8b949e;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stTextArea textarea {
        background-color: #161b22;
        color: #e6edf3;
        border: 1px solid #30363d;
        border-radius: 8px;
    }
    .metric-card {
        background: rgba(22, 27, 34, 0.8);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        backdrop-filter: blur(10px);
        margin-bottom: 15px;
    }
    .metric-title {
        font-size: 1.1rem;
        color: #8b949e;
        margin-bottom: 10px;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #58a6ff;
    }
    .trait-badge {
        display: inline-block;
        padding: 5px 15px;
        margin: 5px;
        border-radius: 20px;
        background: linear-gradient(90deg, #8a2be2, #4b0082);
        font-weight: bold;
        color: white;
        box-shadow: 0 2px 8px rgba(138,43,226, 0.4);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def init_models():
    return {
        "summarizer": load_summarizer(),
        "sentiment": load_sentiment_analyzer(),
        "ner": load_ner_model(),
        "embedder": load_embedding_model()
    }

st.markdown('<div class="main-header">Deep Text Analytics</span></div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Upload or paste your conversation for an AI-powered behavioral dive</div>', unsafe_allow_html=True)

with st.spinner("Initializing AI engines... (Might take a moment on first load)"):
    models = init_models()

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### 📥 Input Panel")
    input_text = st.text_area("Paste your text here...", height=300)
    uploaded_file = st.file_uploader("Or upload a .txt log file", type=["txt"])
    
    if uploaded_file is not None:
        file_text = uploaded_file.read().decode("utf-8")
        # Combine uploaded text with anything typed in the text box
        input_text = file_text + "\n" + input_text

    analyze_btn = st.button("🚀 Analyze Now", use_container_width=True)

if analyze_btn and input_text.strip():
    with st.spinner("Crunching data through transformer layers..."):
        # 1. Preprocessing
        chunks = chunk_text(input_text, max_words=300)
        
        # 2. Analytics
        sum_text = []
        for ch in chunks[:2]: # limit for speed
            sum_text.append(summarize_text(models['summarizer'], ch))
        final_summary = " ".join(sum_text)
        
        keywords = extract_keywords(input_text, top_n=8)
        
        overall_sent = analyze_sentiment(models['sentiment'], input_text)
        time_series = track_sentiment_over_time(models['sentiment'], input_text)
        
        entities = extract_entities(models['ner'], input_text)
        
        behavior = analyze_behavior(input_text)
        
        traits = detect_personality(overall_sent, behavior, keywords)

    with col2:
        st.markdown("### 🧩 Analytical Insights")
        
        # Row 1: Summary & Personality
        r1c1, r1c2 = st.columns(2)
        with r1c1:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.markdown("<div class='metric-title'>📝 Abstractive Summary</div>", unsafe_allow_html=True)
            st.write(final_summary)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with r1c2:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.markdown("<div class='metric-title'>🎭 Personality & Tone</div>", unsafe_allow_html=True)
            traits_html = "".join([f"<span class='trait-badge'>{t}</span>" for t in traits])
            st.markdown(traits_html, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        # Row 2: Metrics
        r2c1, r2c2, r2c3 = st.columns(3)
        with r2c1:
            st.markdown(f"<div class='metric-card'><div class='metric-title'>Sentiment Score</div><div class='metric-value' style='color: {'#3fb950' if overall_sent['label']=='POSITIVE' else '#f85149'}'>{overall_sent['label']} ({(overall_sent['score']*100):.1f}%)</div></div>", unsafe_allow_html=True)
        with r2c2:
            st.markdown(f"<div class='metric-card'><div class='metric-title'>Sentences</div><div class='metric-value'>{behavior['total_sentences']}</div></div>", unsafe_allow_html=True)
        with r2c3:
            st.markdown(f"<div class='metric-card'><div class='metric-title'>Avg Length</div><div class='metric-value'>{behavior['avg_words_per_sentence']} <span style='font-size:0.8rem;color:#8b949e'>words/sent</span></div></div>", unsafe_allow_html=True)
            
        # Row 3: Keywords & Entities
        r3c1, r3c2 = st.columns(2)
        with r3c1:
            with st.expander("🔑 Contextual Keywords", expanded=True):
                st.write(", ".join(keywords))
        with r3c2:
            with st.expander("📇 Named Entities", expanded=True):
                for k, v in entities.items():
                    if v:
                        st.write(f"**{k}**: {', '.join(v)}")
                        
        # Row 4: Sentiment over time
        st.markdown("#### 📉 Emotional Intensity Map")
        if len(time_series) > 1:
            chart_data = pd.DataFrame(time_series, columns=["Sentiment Score"])
            st.line_chart(chart_data, color="#8a2be2")
        else:
            st.info("Not enough sentences to graph sentiment timeline.")

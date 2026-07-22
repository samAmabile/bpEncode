import sys
import os 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import bpEncode
import nltk
from nltk.corpus import brown
import pandas as pd 
import base64

def set_background(jpeg):
    with open(jpeg, "rb") as f:
        encoded_str = base64.b64encode(f.read()).decode()

    css = f"""
    <style>
    .stApp {{
        background-image: linear-gradient(rgba(0, 0, 0, 0.8), rgba(0, 0, 0, 0.8)), url("data:image/jpeg;base64,{encoded_str}");
        background-size: cover;
        backround-position: center;
        background-repeat: no-repeat;
    }}

    section[data-textid="stSidebar"] {{
        background-color: rgba(15, 15, 15, 0.85);
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

st.set_page_config(
    page_title="BPE Tokenizer Demo",
    page_icon="👾",
    layout="centered"
)

set_background("images/owl_scowl.jpeg")

nltk.download('brown')

def build_training_corpus(limit=10000):
    cur_words = 0
    text = []

    for sentence in brown.sents():
        sent_str = " ".join(sentence)
        num_words = len(sentence)

        if cur_words + num_words <= limit:
            text.append(sent_str)
            cur_words += num_words
        else:
            break

    corpus = '\n'.join(text).lower()

    return corpus


@st.cache_resource
def load_tokenizer(vocab_limit):

    tokenizer = bpEncode.BPE()

    corpus = build_training_corpus()

    tokenizer.setVocabSize(vocab_limit)

    modelPath = f"bpe_models/brown_{vocab_limit}.bin"

    try: 
        tokenizer.load(modelPath)
        #print("Loaded pre-trained model...")

    except:
        #print("Training new model on Brown Corpus...")
        corpus = build_training_corpus()
        dictionary = tokenizer.fitWords(corpus)
        tokenizer.save(modelPath)
    
    return tokenizer

st.title("Byte-pair Encoding (BPE) tokenizer")
st.markdown("See how BPE breaks text up into tokens")

st.sidebar.header("Configuration")
vocab_limit = st.sidebar.selectbox(
        "Vocabulary Limit",
        options=[500, 1000, 5000, 10000],
        index=1 
)

with st.spinner(f"Loading {vocab_limit} token limit Brown corpus into tokenizer..."):
    tokenizer = load_tokenizer(vocab_limit)

st.sidebar.success("Model Loaded Succesfully")

sample = st.text_input("Enter text to tokenize:", value="The quick brown mink mourned the folly of mankind")

if sample:
    tokens = tokenizer.tokenize(sample)

    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Total Tokens", value=len(tokens))
    with col2:
        words = len(sample.split())
        ratio = round(len(tokens) / max(1, words), 2)
        st.metric(label="Tokens / Word", value=ratio)

    st.divider()

    st.subheader("Breakdown")
    st.markdown("**Byte-Pair token IDs:**")
    tokenstr = " ".join(str(t) for t in tokens)
    st.code(tokenstr)

    decoded = [] 
    for token in tokens:
        chunk = tokenizer.decode([token])
        decoded.append(chunk)

    st.markdown("**Sub-Word tokens:**")
    st.code(" ".join(decoded), language="text")
    
    st.divider()

    st.subheader("Token Density Map")

    words = sample.split()
    token_counts = []

    for word in words:
        word_tokens = tokenizer.tokenize(word)
        token_counts.append(len(word_tokens))

    df = pd.DataFrame({
        "word": words,
        "tokens": token_counts
    })

    chart_data = df.set_index("word")

    st.line_chart(chart_data)



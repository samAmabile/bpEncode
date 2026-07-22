import sys
import os 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import bpEncode
import nltk
from nltk.corpus import brown
import pandas as pd 
import base64
import matplotlib.pyplot as plt 
import seaborn as sns 


def set_background(jpg):
    with open(jpg, "rb") as f:
        encoded_str = base64.b64encode(f.read()).decode()

    css = f"""
    <style>
    .stApp {{
        background-image: linear-gradient(rgba(20, 24, 33, 0.003), rgba(20, 24, 33, 0.003)), url("data:image/jpg;base64,{encoded_str}");
        background-size: cover;
        backround-position: center;
        background-repeat: no-repeat;
    }}

    .stMain, div.block-container {{
        backround-color: rgba(20, 24, 33, 0.2);
        border-radius: 12px;
        padding: 2rem;
    }}

    section[data-testid="stSidebar"] {{
        background-color: rgba(15, 18, 25, 0.2);
    }}

    h1, h2, h3, h4, h5, h6, p, span, label, div[data-testid="stMarkdownContainer"] {{
        color: #38bdf8 !important;
        text-shadow: 0px 2px 4px rgba(0, 0, 0, 0.9), 0px 0px 3px rgba(0, 0, 0, 0.8) !important;
    }}

    input {{
        color: #38bdf8 !important;
        background-color: rgba(20, 25, 35, 0.8) !important;
        border: 1px solid rgba(56, 189, 248, 0.3) !important;
        text-shadow: 0px 1px 2px rgba(0, 0, 0, 0.8);
    }}

    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

st.set_page_config(
    page_title="BPE Tokenizer Demo",
    page_icon="👾",
    layout="centered"
)

set_background("images/owl_scowl.jpg")

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

    fig, ax = plt.subplots(figsize=(10, 4))

    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')

    sns.lineplot(data=df, x="word", y="tokens", marker="o", ax=ax, color="#ff4b4b", linewidth=2.5)

    ax.set_xlabel("Word", color="white")
    ax.set_ylabel("Tokens", color="white")
    ax.set_title("Tokens per word in your text", color="red")

    ax.set_ylim(0, 10)

    plt.xticks(rotation=30, ha="right", color="white")
    plt.yticks(color="white")

    ax.tick_params(colors="white")
    for _ in ax.spines.values():
        _.set_edgecolor('#555555')

    fig.tight_layout()
    st.pyplot(fig)




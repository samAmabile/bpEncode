import sys
import os 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import bpEncode
import nltk
from nltk.corpus import brown

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
def load_tokenizer():

    tokenizer = bpEncode.BPE()

    corpus = build_training_corpus()

    try: 
        tokenizer.load("bpe_models/brown.bin")
        print("Loaded pre-trained model...")

    except:
        print("Training new model on Brown Corpus...")
        corpus = build_training_corpus()
        dictionary = tokenizer.fitWords(corpus)
        tokenizer.save("bpe_models/brown.bin")
    
    return tokenizer


st.title("Byte-pair Encoding (BPE) tokenizer")
st.markdown("See how BPE breaks text up into tokens")

with st.spinner("Loading Brown corpus into tokenizer..."):
    tokenizer = load_tokenizer()

st.success("Ready!")

sample = st.text_input("Enter text to tokenize:", value="The quick brown mink mourned the folly of mankind")

if sample:
    tokens = tokenizer.tokenize(sample)

    st.subheader("Breakdown")
    st.markdown("**Byte-Pair token IDs:**")
    tokenstr = " ".join(str(t) for t in tokens)
    st.code(tokens)

    decoded = [] 
    for token in tokens:
        chunk = tokenizer.decode([token])
        decoded.append(chunk)

    st.markdown("**Sub-Word tokens:**")
    st.code(" ".join(decoded), language="text")




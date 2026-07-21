import bpEncode
import pandas as pd 
import numpy as np 
from pathlib import Path
import importlib
import string

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SRC = ROOT / "src"

importlib.reload(bpEncode)

english_corpus = open(DATA / "english.txt", 'r', encoding='utf-8').read()
raramuri_corpus = open(DATA / "raramuri.txt", 'r', encoding='utf-8').read() 
nahuatl_corpus = open(DATA / "nahuatl.txt", 'r', encoding='utf-8').read() 

size_limit = 900000

english_corpus = ' '.join(english_corpus.split()[:size_limit])
raramuri_corpus = ' '.join(raramuri_corpus.split()[:size_limit])
nahuatl_corpus = ' '.join(nahuatl_corpus.split()[:size_limit])

def cleanText(text):

    punctuation = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    digits = str.maketrans('', '', string.digits)

    text = text.translate(digits)
    text = text.translate(punctuation)

    return ' '.join(text.lower().split())

def makeRepresentativeSample(text, size=120000): 
    words = text.strip().split()
    N = len(words)
    midpoint = N // 2
    slice = size // 3
    first = words[:slice]
    middle = words[midpoint - (slice // 2) : midpoint + (slice // 2)]
    end = words[:-slice]
    sample = first + middle + end

    return ' '.join(sample)

def get_random_sample(text, size):
    idx = np.random.randint(0, len(text) - size)
    return text[idx : idx + size]

def compile_data(corpus, language, vocab_limit, n_samples=10, batch_size=5000):
    words = corpus.split()
    tokenizer = bpEncode.BPE()
    tokenizer.setVocabSize(vocab_limit)
    

    sample = makeRepresentativeSample(corpus)
    dictionary = tokenizer.fitWords(sample)

    data = []
    for i in range(n_samples):
        batch = get_random_sample(words, batch_size)
        tokens = tokenizer.tokenize(' '.join(batch))

        data.append({
            "language": language,
            "bpe vocab limit": vocab_limit,
            "tokens_per_word": len(tokens) / len(batch)
        })
    df = pd.DataFrame(data)
    return df


english_corpus = cleanText(english_corpus)
raramuri_corpus = cleanText(raramuri_corpus)
nahuatl_corpus = cleanText(nahuatl_corpus)

datasets_5c = []
datasets_1k = []
datasets_5k = []
datasets_10k = []

for language, corpus in [("english", english_corpus), ("raramuri", raramuri_corpus), ("nahuatl", nahuatl_corpus)]:
    datasets_5c.append(compile_data(corpus, language, 500))
    datasets_1k.append(compile_data(corpus, language, 1000))
    datasets_5k.append(compile_data(corpus, language, 5000))
    datasets_10k.append(compile_data(corpus, language, 10000))

df_5c = pd.concat(datasets_5c, ignore_index=True)
df_1k = pd.concat(datasets_1k, ignore_index=True)
df_5k = pd.concat(datasets_5k, ignore_index=True)
df_10k = pd.concat(datasets_10k, ignore_index=True)


df = pd.concat([df_5c, df_1k, df_5k, df_10k])

df.to_csv(DATA / "bootstrapped_exp.csv")







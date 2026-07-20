import pandas as pd 
import numpy as np 
import bpEncode 
from bpEncode import BPE 
import re 
import string
import importlib 

importlib.reload(bpEncode)


import matplotlib.pyplot as plt 
import seaborn as sns 

#TODO: resolve data paths to root/data/filename:
english_corpus = open("../data/english.txt", 'r', encoding='utf-8').read()
raramuri_corpus = open("../data/raramuri.txt", 'r', encoding='utf-8').read() 
nahuatl_corpus = open("../data/nahuatl.txt", 'r', encoding='utf-8').read() 

"""
size_limit = 900000

english_corpus = ' '.join(english_corpus.split()[:size_limit])
raramuri_corpus = ' '.join(raramuri_corpus.split()[:size_limit])
nahuatl_corpus = ' '.join(nahuatl_corpus.split()[:size_limit])
"""
def cleanText(text):

    punctuation = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    digits = str.maketrans('', '', string.digits)

    text = text.translate(digits)
    text = text.translate(punctuation)

    return ' '.join(text.lower().split())

def makeRepresentativeSample(text, size=300000): 
    words = text.strip().split()
    N = len(words)
    midpoint = N // 2
    slice = size // 3
    first = words[:slice]
    middle = words[midpoint - (slice // 2) : midpoint + (slice // 2)]
    end = words[:-slice]
    sample = first + middle + end

    return ' '.join(sample)


def batchText(text, sizes=[1000, 2000, 4000, 8000, 16000, 32000, 64000]):
    batches = []
    words = text.strip().split()
    i = 0
    for size in sizes:
        batch = words[i : i + size] 
        batches.append(' '.join(batch))
        i = i + size + 1
        if i >= len(text): break
    
    return batches

    

english_corpus = cleanText(english_corpus)
raramuri_corpus = cleanText(raramuri_corpus)
nahuatl_corpus = cleanText(nahuatl_corpus)

corpora = [english_corpus, raramuri_corpus, nahuatl_corpus] 

vocab_sizes = [200, 500, 1000, 5000, 10000, 15000]

data = []

def compile_data(corpus, language, size):
    tokenizer = BPE()
    tokenizer.setVocabSize(size)

    repSample = makeRepresentativeSample(corpus) 
    tokenizer.fitWords(repSample)

    batches = batchText(corpus)

    data = []

    for batch in batches:
        tokens = tokenizer.tokenize(batch)

        num_words = len(batch.split())

        tokens_per_word = len(tokens) / num_words
        words = batch.split() 
        individual_word_tokens = []
        individual_word_characters = []
        for word in words:
            word_tokens = tokenizer.tokenize(word)
            individual_word_tokens.append(len(word_tokens))
            individual_word_characters.append(len(word))

        word_token_counts = np.array(individual_word_tokens)
        word_char_counts = np.array(individual_word_characters)
        batchData = {
            "language": language,
            "bpe vocab limit": size,
            "num tokens": len(tokens),
            "num words": num_words, 
            "tokens : word": tokens_per_word,
            "word lengths": word_char_counts,
            "word tokens": word_token_counts,
            "mean word len": np.mean(word_char_counts),
            "mean tokens per word": np.mean(word_token_counts)
        }

        data.append(batchData)
    
    tokenizer.reset()
    return data

master_data = []

languages = ["english", "raramuri", "nahuatl"]

for language, corpus in zip(languages, corpora):
    for size in vocab_sizes:
        size_data = compile_data(corpus, language, size)
        master_data.extend(size_data)

df = pd.DataFrame(master_data)

df.to_csv("proposal_data_demo_v9.csv", index=False)


stats_df = df.groupby(['language', 'bpe vocab limit'])['tokens : word'].agg(['mean', 'std'])

stats_df.to_csv("proposal_stats_summary_v9.csv")





    


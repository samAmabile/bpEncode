import bpEncode
from bpEncode import BPE
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import re 
from collections import Counter
import importlib
import string

importlib.reload(bpEncode)

def batchText(text, batchSize=5000):
    batches = []
    for i in range(0, (len(text) - batchSize), batchSize):
        batch = text[i:i + batchSize]
        batches.append(batch)

    return batches

def cleanText(text):

    punctuation = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    digits = str.maketrans('', '', string.digits)

    text = text.translate(digits)
    text = text.translate(punctuation)

    return ' '.join(text.lower().split())

def makeSample(text, size=20000): 
    words = text.strip().split()
    sample = words[:size]
    return ' '.join(sample)

english_txt_raw = ""
with open("english.txt", 'r', encoding='utf-8') as f:
    for line in f:
        if line:
            clean = re.sub(r'^\d+$', '', line)
            if clean:
                english_txt_raw += line



nahuatl_txt_raw = ""
with open("nahuatl.txt", 'r', encoding='utf-8') as f1:
    nahuatl_txt_raw = f1.read(); 

raramuri_txt_raw = ""
with open("raramuri.txt", 'r', encoding='utf-8') as f2:
    raramuri_txt_raw = f2.read()

nahuatl_txt = cleanText(nahuatl_txt_raw)
raramuri_txt = cleanText(raramuri_txt_raw)
english_txt = cleanText(english_txt_raw)

nahuatl_sample = makeSample(nahuatl_txt)
raramuri_sample = makeSample(raramuri_txt)
english_sample = makeSample(english_txt) 

nah_tokenizer = BPE()
rar_tokenizer = BPE()
eng_tokenizer = BPE()

corpus = nahuatl_txt + raramuri_txt + english_txt
#corpus_sample = nahuatl_txt[:50000]+nahuatl_txt[:-50000] + raramuri_txt[:50000]+raramuri_txt[:-50000] + english_txt[:50000]+english_txt[:-50000]
nah_vocab = nah_tokenizer.fitWords(nahuatl_sample)
rar_vocab = rar_tokenizer.fitWords(raramuri_sample)
eng_vocab = eng_tokenizer.fitWords(english_sample)

vocab = list(nah_vocab.keys()) + list(rar_vocab.keys()) + list(eng_vocab.keys())
#vocab = tokenizer.fitWords(corpus_sample)

english_tokens = eng_tokenizer.tokenize(english_txt)
nahuatl_tokens = nah_tokenizer.tokenize(nahuatl_txt)
raramuri_tokens = rar_tokenizer.tokenize(raramuri_txt)
total_tokens = len(nahuatl_tokens) + len(raramuri_tokens) + len(english_tokens)

common_tokens = [x for x in nahuatl_tokens if x in raramuri_tokens] #raranmuri and nhuatl only 
all_common_tokens = [x for x in english_tokens if x in common_tokens]

print("First few tokens from each: ")
print("Nahuatl: ")
print(nahuatl_tokens[:10], "...")
print("Raramuri: ")
print(raramuri_tokens[:10], "...")
print("English: ")
print(english_tokens[:10], "...")
print("Number of shared byte-pair tokens (raramuri and nahuatl): ", len(common_tokens))



def getMATTR(tokens, window=500):
    ratios = []
    for i in range(len(tokens)-window): 
        cur_tokens = tokens[i:i+window]
        cur_types = set(cur_tokens)
        cur_TTR = float(len(cur_types) / len(cur_tokens))
        ratios.append(cur_TTR)

    return np.mean(ratios)

def getFreqDist(tokens):
    return Counter(tokens)
def get_NGrams(text, sizes = [2, 3]):
    ngrams = Counter()
    
    maxLen = max(sizes)
    for i in range(len(text) - maxLen):
        for window in sizes:
            ngram = text[i:i+window]
            if ngram in ngrams:
                ngrams[ngram] += 1
            else:
                ngrams[ngram] = 1

    return ngrams

def get_ngram_morphemes(wordlist, sizes=[2, 3]):
    morphemes = Counter()

    for word in wordlist:

        padded_word = f"^{word}$"

        for size in sizes:
            for i in range(len(padded_word) - size + 1):
                ngram = padded_word[i : i + size]
                morphemes[ngram] += 1 

    unique = len(morphemes)
    total = morphemes.total()
    entropy = shannon_entropy(morphemes)

    return unique, entropy, morphemes


def shannon_entropy(fdist):
    total = sum(fdist.values())
    entropy = 0; 
    for count in fdist.values():
        p = count / total
        if p > 0:
            entropy -= p * np.log2(p)

    return entropy

    

#words split on white space:
all_words = corpus.strip().split()
nahuatl_words = nahuatl_txt.strip().split()
raramuri_words = raramuri_txt.strip().split()
english_words = english_txt.strip().split()

#global TTR , nahuatl TTR and raramuri TTR:
corpus_TTR = float(len(set(all_words)) / len(all_words))
nahuatl_TTR = float(len(set(nahuatl_words)) / len(nahuatl_words))
raramuri_TTR = float(len(set(raramuri_words)) / len(raramuri_words))
english_TTR = float(len(set(english_words)) / len(english_words))

corpus_mean_word_len = float(np.mean(np.array([len(x) for x in all_words])))
nahuatl_mean_word_len = float(np.mean(np.array([len(x) for x in nahuatl_words])))
raramuri_mean_word_len = float(np.mean(np.array([len(x) for x in raramuri_words])))
english_mean_word_len = float(np.mean(np.array([len(x) for x in english_words])))

#Moving Average TTRs:
corpus_MATTR = getMATTR(all_words)
nahuatl_MATTR = getMATTR(nahuatl_words)
raramuri_MATTR = getMATTR(raramuri_words)
english_MATTR = getMATTR(english_words)

#frequency distribution of Ngrams:
corpus_ngrams = get_NGrams(corpus)
nahuatl_ngrams = get_NGrams(nahuatl_txt)
raramuri_ngrams = get_NGrams(raramuri_txt)
english_ngrams = get_NGrams(english_txt)

#frequency distribution of words:
corpus_fdist = getFreqDist(all_words)
nahuatl_fdist = getFreqDist(nahuatl_words)
raramuri_fdist = getFreqDist(raramuri_words)
english_fdist = getFreqDist(english_words)

#frequency distribution of byte-pair tokens;
nahuatl_tokens_fdist = getFreqDist(nahuatl_tokens)
raramuri_tokens_fdist = getFreqDist(raramuri_tokens)
english_tokens_fdist = getFreqDist(english_tokens)

def parse_morphemes(words):
    unique_morphemes, morpheme_entropy, morpheme_fdist = get_ngram_morphemes(words)
    return unique_morphemes, morpheme_entropy, morpheme_fdist

nahuatl_morpheme_data = parse_morphemes(nahuatl_words)
raramuri_morpheme_data = parse_morphemes(raramuri_words)
english_morpheme_data = parse_morphemes(english_words)


def agglutination_proxy(common_morph, rare_morph, ngrams):
    total_ngrams = sum(ngrams.values())

    if total_ngrams == 0: return 0, 0 

    common_morpheme_score = sum(ngrams[n] for n in common_morph) / total_ngrams
    rare_morpheme_score = sum(ngrams[n] for n in rare_morph) / total_ngrams

    return common_morpheme_score, rare_morpheme_score

def compile_global_stats(text, tokenizer):
    cleanText = re.sub(r'([.,;:\'\"\(\)*!?])', '', text) 
    words = cleanText.lower().strip().split()
    tokens = tokenizer.tokenize(text)
    TTR = len(set(words)) / len(words)
    MATTR = getMATTR(words)
    mean_word_len = np.mean(np.array([len(w) for w in words]))
    fdist = getFreqDist(words)
    morpheme_data = parse_morphemes(words)
    num_unique_morphemes = morpheme_data[0]
    morpheme_entropy = morpheme_data[1]
    morpheme_fdist = morpheme_data[2]
    shan_ent = shannon_entropy(fdist)
    hapaxes = [w for w, count in fdist.items() if count == 1] 
    hapax_ratio = len(hapaxes) / len(words)
    

corpus_data = {
        "language": "corpus",
        "total characters": len(corpus),
        "total words": len(all_words),
        "total bpe tokens": len(raramuri_tokens) + len(nahuatl_tokens) + len(english_tokens),
        "unique tokens": len(list(vocab)),
        "unique morphemes": nahuatl_morpheme_data[0] + raramuri_morpheme_data[0],
        "Avg Word Length": corpus_mean_word_len,
        "hapax ratio": float(len([w for w, count in corpus_fdist.items() if count == 1]) / len(all_words)),
        "agglutination": float(total_tokens / len(all_words)),
        "shannon entropy": shannon_entropy(corpus_fdist),
        "global TTR": corpus_TTR,
        "MATTR": corpus_MATTR,
        "text": corpus
}


nahuatl_data = {
        "language": "nahuatl",
        "total characters": len(nahuatl_txt),
        "total words": len(nahuatl_words),
        "total bpe tokens": len(nahuatl_tokens),
        "unique tokens": len(set(nahuatl_tokens)),
        "unique morphemes": nahuatl_morpheme_data[0],
        "Avg Word Length": nahuatl_mean_word_len,
        "hapax ratio": float(len([w for w, count in nahuatl_fdist.items() if count == 1]) / len(nahuatl_words)),
        "agglutination": float(len(nahuatl_tokens) / len(nahuatl_words)),
        "shannon entropy": shannon_entropy(nahuatl_fdist),
        "global TTR": nahuatl_TTR,
        "MATTR": nahuatl_MATTR,
        "text": nahuatl_txt
}


raramuri_data = {
        "language": "raramuri",
        "total characters": len(raramuri_txt),
        "total words": len(raramuri_words),
        "total bpe tokens": len(raramuri_tokens),
        "unique tokens": len(set(raramuri_tokens)),
        "unique morphemes": raramuri_morpheme_data[0],
        "Avg Word Length": raramuri_mean_word_len,
        "hapax ratio": float(len([w for w, count in raramuri_fdist.items() if count == 1]) / len(raramuri_words)),
        "agglutination": float(len(raramuri_tokens) / len(raramuri_words)),
        "shannon entropy": shannon_entropy(raramuri_fdist),
        "global TTR": raramuri_TTR,
        "MATTR": raramuri_MATTR,
        "text": raramuri_txt
}


english_data = {
        "language": "english",
        "total characters": len(english_txt),
        "total words": len(english_words),
        "total bpe tokens": len(english_tokens),
        "unique tokens": len(set(english_tokens)),
        "unique morphemes": english_morpheme_data[0],
        "Avg Word Length": english_mean_word_len,
        "hapax ratio": float(len([w for w, count in english_fdist.items() if count == 1]) / len(english_words)),
        "agglutination": float(len(english_tokens) / len(english_words)),
        "shannon entropy": shannon_entropy(english_fdist),
        "global TTR": english_TTR,
        "MATTR": english_MATTR,
        "text": english_txt
}


        
data = [corpus_data, nahuatl_data, raramuri_data, english_data]

df = pd.DataFrame(data)

df.to_csv("raramuri_nahuatl_corpus.csv")

global_metrics = df.drop(columns=["text"])

metrics_to_plot = ["language", "total words", "total bpe tokens", "unique tokens"]

plot_df = global_metrics[metrics_to_plot]

plot_df_melt = plot_df.melt(id_vars="language", var_name="metric", value_name="count")
graph = sns.catplot(
        data=plot_df_melt,
        x="language",
        y="count",
        hue="metric",
        kind="bar",
        palette="viridis",
        height=6,
        aspect=1.5 
)

graph.set_axis_labels("Language", "Count (Log Scale)")
plt.title("Corpus Summary Statistics") 
graph.set(yscale="log")
plt.show()
plt.savefig("rara_nahua_global.png", dpi=300)



nahuatl_batches = batchText(nahuatl_txt)
raramuri_batches = batchText(raramuri_txt)
english_batches = batchText(english_txt)

def getBatchData(batches, language, morphemes, tokenizer):
    dataset = []
    for batch in batches:
        
        cleanBatch = ""
        for symbol in "!?,.;:()":
            cleanBatch = batch.replace(symbol, "")
        
        cleanBatch = re.sub(r'\d+', '', cleanBatch)
        tokens = tokenizer.tokenize(batch)
        words = cleanBatch.strip().split()


        total_morphemes = sum(morphemes.values())
        common_morphemes = {m for m, count in morphemes.most_common(20)}
        rare_threshold = total_morphemes * 0.001
        rare_morphemes = {m for m, count in morphemes.items() if count <= rare_threshold}
        
        ngrams = get_NGrams(cleanBatch)
        common_score, rare_score = agglutination_proxy(common_morphemes, rare_morphemes, ngrams)

        avg_word_len = float(np.mean([len(w) for w in words]))
        ngrams = get_NGrams(cleanBatch)
        fdist = getFreqDist(words)
        token_fdist = getFreqDist(tokens)
        stop_words = {w for w, count in fdist.most_common(100)}
        content_words = [word for word in words if word not in stop_words]
        ttr = float(len(set(words)) / len(words))
        mattr = getMATTR(batch, 200)


        data = {
                "language": language,
                "num characters": len(batch),
                "num words": len(words),
                "num ngrams": ngrams.total(),
                "num bpe tokens": len(tokens),
                "avg word length": avg_word_len,
                "word hapax ratio": float(len([w for w, count in fdist.items() if count == 1]) / len(words)),
                "ngram hapax ratio": float(len([g for g, count in ngrams.items() if count == 1]) / ngrams.total()),
                "token hapax ratio": float(len([t for t, count in token_fdist.items() if count == 1]) / len(tokens)),
                "agglutination": float(len(tokens) / len(words)),
                "common morpheme score": common_score,
                "rare morpheme score": rare_score,
                "word entropy": shannon_entropy(fdist),
                "ngram entropy": shannon_entropy(ngrams),
                "token entropy": shannon_entropy(token_fdist),
                "lexical density": float(len(content_words) / len(words)),
                "ttr": ttr,
                "mattr": mattr,
                "text": cleanBatch,
                "tokens": tokens
        }

        dataset.append(data)

    return dataset

nahuatl_dataset = getBatchData(nahuatl_batches, "nahuatl", nahuatl_morpheme_data[2], nah_tokenizer)
raramuri_dataset = getBatchData(raramuri_batches, "raramuri", raramuri_morpheme_data[2], rar_tokenizer)
english_dataset = getBatchData(english_batches, "english", english_morpheme_data[2], eng_tokenizer)

master_dataset = nahuatl_dataset + raramuri_dataset + english_dataset

master_df = pd.DataFrame(master_dataset) 

master_df.to_csv("batch_level_corpus.csv") 















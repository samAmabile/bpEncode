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



print("BPE Tokenizer for English prepared for demo")


while(True):
    
    sample = input("Enter text to tokenize, or enter 'x' to exit: ")

    if sample.lower() == 'x': 
        break

    tokens = tokenizer.tokenize(sample) 

    print("\nByte-Pair tokens: ", tokens)

    decoded = [] 

    for token in tokens: 
        word_chunk = tokenizer.decode([token])
        decoded.append(word_chunk)

    print("\nText tokens: ")
    print(" ".join(decoded))

    

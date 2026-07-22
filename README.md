# Byte-Pair encoding as a proxy for Morphological Complexity

Using custom BPE algorithm to measure the ability of algorithmic tokenization to quantify morphological complexity in low resource languages

## Byte-Pair Encoding Algorithm 

BPE scans text and merges the most frequent adjacent characters until a maximum limit of unique tokens is reached. This process could be used to analyze languages with less settled morphology and provide evidence for the analytic or agglutinative quality of the language. 

See how BPE works here: 

[BPE Demo](https://tokenz.streamlit.app/)

### Details 

- **bpEncode**: bpe.cpp and bpe.hpp, the actual BPE logic, written in CPP so it can be optimized excessively. Linked to python with cmake and mypyproject.toml for use in python. 
- **data_prep.py**: Script to gather and analyze data from English, Nahuatl, and Raramuri
- **visualize.py**: Script to plot csv data for exploratory analysis into data relationships




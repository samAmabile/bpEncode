import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from pathlib import Path 

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SRC = ROOT / "src" 
IMG = ROOT / "images"

df = pd.read_csv(DATA / "bootstrapped_exp.csv")

mean_fertility = df.groupby(["bpe vocab limit", "language"], as_index=False)["tokens_per_word"].mean()

plt.figure(figsize=(10, 6))
sns.barplot(
        data=mean_fertility,
        x="bpe vocab limit",
        y="tokens_per_word",
        hue="language",
)

plt.title("Token Fertility v. BPE Vocab Limit")
plt.xlabel("BPE Max Unique Tokens")
plt.ylabel("Mean Tokens per Word")
plt.grid(True, linestyle="--", alpha=0.6)
plt.savefig(IMG / "vocab_limit_fertility_bar.png", dpi=300)
plt.show()





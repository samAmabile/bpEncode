import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
from pathlib import Path 

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SRC = ROOT / "src"
IMG = ROOT / "images"

df = pd.read_csv(DATA / "new_data_v1.csv") 

sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 4, figsize=(22, 5))
#1 stability:
sns.lineplot(data=df, x="num words", y="tokens : word", hue="language", ax=axes[0])
axes[0].set_title("Morphological stability")

# 2. Morphological Distribution (Signature)
sns.scatterplot(data=df, x="num words", y="mean tokens per word", hue="language", ax=axes[1])
axes[1].set_title("Morphological Signature")

# 3. Distribution Plot (Histogram of Token/Word Ratios)
sns.histplot(data=df, x="mean tokens per word", hue="language", kde=True, ax=axes[2])
axes[2].set_title("Distribution of Fertility")

# 4. Num Words to Num Tokens (Global Fertility Rays)
sns.regplot(data=df, x="num words", y="num tokens", scatter=True, ax=axes[3])
axes[3].set_title("Global Fertility Rays")

plt.tight_layout()
plt.savefig(IMG / "new_data_plots.pdf")
plt.show()


#Distribution plot: 



#Num Words to Num Tokens 

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.manifold import TSNE
vowels = "aeiou"
consonants = "bcdfghjklmnpqrstvwxyz"
special = "."

sns.set_theme(style="darkgrid", context="talk", font_scale=0.9)
def plot_embeddings(data, rev_lkp_tbl, turn):
    data_12d      = data.detach().cpu().numpy()
    tsne          = TSNE(n_components=2, perplexity=5, random_state=42)
    embeddings_2d = tsne.fit_transform(data_12d)
    categories = []
    for i in range(data.shape[0]):
        char = rev_lkp_tbl[i]
        if char in vowels:
            categories.append("Vowel")
        elif char in consonants:
            categories.append("Consonant")
        elif char in special:
            categories.append("Special")

    plt.figure(figsize=(10, 10))
    ax2 = sns.scatterplot(
        x=embeddings_2d[:, 0],
        y=embeddings_2d[:, 1],
        hue=categories,
        palette="husl",
        s=600,
        edgecolor="white",
        linewidth=1.5,
        alpha=0.9
    )
    for i in range(data.shape[0]):
        ax2.text(
            embeddings_2d[i, 0],
            embeddings_2d[i, 1],
            rev_lkp_tbl[i],
            ha="center",
            va="center",
            color="white",
            fontsize=12,
            fontweight="bold"
        )

    ax2.set_title("t-SNE of Character Embeddings Color-Coded by Class", pad=20, fontweight="bold")
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.savefig(f"tsne_character_embeddings_{turn}_training_test.jpeg", dpi=300)

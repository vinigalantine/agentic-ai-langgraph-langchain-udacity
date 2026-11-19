"""
    Exercise 30 - LangChain and embeddings

    Challenge:  None, just checking out embeddings.
"""

import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from typing import Literal
from langchain_core.embeddings import Embeddings
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.decomposition import PCA

load_dotenv()

def print_spacing():
    print("\n" + "="*100 + "\n")

class EmbeddingsFactory:
    def __init__(self, provider: Literal["OpenAI", "HugginFace"], **kwargs):
        self.provider = provider
        self.kwargs = kwargs

    def create(self) -> Embeddings:
        if self.provider == "OpenAI":
            return OpenAIEmbeddings(**self.kwargs)
        if self.provider == "HugginFace":
            return HuggingFaceEmbeddings(**self.kwargs)

        raise ValueError(f"Unknown embeddings provider: {self.provider}")


sentence_list = [
    "I want to listen to music again",
    "I'm in the mood to hear music once more.",
    "Playstation has been a big part of my childhood",
    "I grew up playing Nintendo games",
    "The place I visited is the same as before",
    "The destination I returned to hasn’t changed over the years",
]

embeddings = EmbeddingsFactory(
    provider="HugginFace",
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": False},
).create()

embeddings = EmbeddingsFactory(provider="OpenAI").create()

embeddings_list = [embeddings.embed_query(sentence) for sentence in sentence_list]

print("Embedding list len: ", len(embeddings_list))

print_spacing()

print("Cool numbers: ", embeddings_list[0][:10])

sentence_embeddings_map = [
    {"sentence": sentence, "embeddings": embeddings} for sentence, embeddings in zip(sentence_list, embeddings_list)
]

def print_similarity(sentence_embeddings_map: list[dict], i1: int = 0, i2: int = 0) -> None:
    s1 = sentence_embeddings_map[i1]["sentence"]
    e1 = sentence_embeddings_map[i1]["embeddings"]
    s2 = sentence_embeddings_map[i2]["sentence"]
    e2 = sentence_embeddings_map[i2]["embeddings"]
    
    print(f"Score: {np.dot(e1,e2):.2f}\n")
    print(f"Sentence {i1}: {s1}\nSentence {i2}: {s2}")

print_spacing()

print_similarity(sentence_embeddings_map, 0, 1)

print_spacing()

print_similarity(sentence_embeddings_map, 2, 3)

print_spacing()

print_similarity(sentence_embeddings_map, 4, 5)

print_spacing()

print_similarity(sentence_embeddings_map, 0, 3)

pca_model = PCA(n_components=2)
pca_model.fit(embeddings_list)
new_values = pca_model.transform(embeddings_list)

print_spacing()

print(f"shape: {new_values.shape}")
print(new_values)

print_spacing()

def plot_2d(x_values, y_values, info_list):
    fig, ax = plt.subplots()
    scatter = ax.scatter(x_values, y_values, alpha=0.5, edgecolors="k", s=40)
    
    ax.set_title("Embeddings Viz in 2D")
    ax.set_xlabel("X_1")
    ax.set_ylabel("X_2")
    
    for i, info in enumerate(info_list):
        ax.annotate(info, (x_values[i], y_values[i]))
        
    plt.savefig("data/embeddings.png", dpi=150, bbox_inches="tight")
    plt.show()
    
plot_2d(new_values[:,0], new_values[:,1], sentence_list)
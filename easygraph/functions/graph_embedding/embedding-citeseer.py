from __future__ import print_function

import csv
import time

import easygraph as eg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from easygraph.functions.community import greedy_modularity_communities
from easygraph.functions.community import modularity
from easygraph.functions.graph_embedding import *
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


warnings.filterwarnings("ignore")

if __name__ == "__main__":
    dataset = eg.CiteseerGraphDataset(
        force_reload=True
    )  # Download CiteseerGraphDataset contained in EasyGraph
    num_classes = dataset.num_classes
    g = dataset[0]
    labels = g.ndata["label"]

    # Graph embedding via DeepWalk
    deepwalk_emb, _ = deepwalk(g, dimensions=32, walk_length=50, num_walks=20)
    #   print(deepwalk_emb, len(deepwalk_emb))

    dw_emb = []
    for i in range(0, len(deepwalk_emb)):
        dw_emb.append(list(deepwalk_emb[i]))
    #   print(len(dw_emb))
    dw_emb = np.array(dw_emb)
    print(dw_emb)

    tsne = TSNE(n_components=2, verbose=1, random_state=0)
    z = tsne.fit_transform(dw_emb)
    # plt.figure(figsize=(6, 6))
    plt.scatter(z[:, 0], z[:, 1], c=labels, s=100)
    plt.savefig("deepwalk_citesee.pdf", bbox_inches="tight")
    plt.show()

    # Graph embedding via Node2Vec
    node2vec_emb, _ = node2vec(
        g, dimensions=32, walk_length=30, num_walks=10, p=0.25, q=4
    )
    #   print(node2vec_emb, len(node2vec_emb))

    n2v_emb = []
    for i in range(0, len(node2vec_emb)):
        n2v_emb.append(list(node2vec_emb[i]))
    #   print(len(n2v_emb))
    n2v_emb = np.array(n2v_emb)
    print(n2v_emb)

    tsne = TSNE(n_components=2, verbose=1, random_state=0)
    z = tsne.fit_transform(n2v_emb)
    # plt.figure(figsize=(6, 6))
    plt.scatter(z[:, 0], z[:, 1], c=labels, s=100)
    plt.savefig("n2v_citesee.pdf", bbox_inches="tight")
    plt.show()

    # Graph embedding via LINE
    model = LINE(g, embedding_size=32, order="second")
    model.train(batch_size=64, epochs=1, verbose=2)
    line_emb = model.get_embeddings()

    l_emb = []
    for i in range(0, len(line_emb)):
        l_emb.append(list(line_emb[i]))
    #   print(len(l_emb))
    l_emb = np.array(l_emb)
    print(l_emb)

    tsne = TSNE(n_components=2, verbose=1, random_state=0)
    z = tsne.fit_transform(l_emb)
    # plt.figure(figsize=(6, 6))
    plt.scatter(z[:, 0], z[:, 1], c=labels, s=100)
    plt.savefig("line_citesee.pdf", bbox_inches="tight")
    plt.show()

    # Graph embedding via SDNE
    model = SDNE(g, hidden_size=[128, 32])
    model.train(batch_size=32, epochs=40, verbose=2)
    sdne_emb = model.get_embeddings()

    sd_emb = []
    for i in range(0, len(sdne_emb)):
        sd_emb.append(list(sdne_emb[i]))
    #   print(len(sd_emb))
    sd_emb = np.array(sd_emb)
    print(sd_emb)

    tsne = TSNE(n_components=2, verbose=1, random_state=0)
    z = tsne.fit_transform(sd_emb)
    #   plt.figure(figsize=(6, 6))
    plt.scatter(z[:, 0], z[:, 1], c=labels, s=100)
    plt.savefig("sdne_citesee.pdf", bbox_inches="tight")
    plt.show()

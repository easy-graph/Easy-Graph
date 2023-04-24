from __future__ import print_function

import argparse
import csv
import time
import warnings

from datetime import datetime

import easygraph as eg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from easygraph.datasets.citation_graph import CiteseerGraphDataset
from easygraph.functions.community import greedy_modularity_communities
from easygraph.functions.community import modularity
from easygraph.functions.graph_embedding import *
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


warnings.filterwarnings("ignore")


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = CiteseerGraphDataset(
        force_reload=True
    )  # Download CiteseerGraphDataset contained in EasyGraph
    num_classes = dataset.num_classes
    g = dataset[0]
    labels = g.ndata["label"]
    print(labels, labels.shape, len(g.nodes))

    print("Graph embedding via DeepWalk...........")
    deepwalk_emb, _ = deepwalk(g, dimensions=128, walk_length=80, num_walks=10)
    # print(deepwalk_emb, len(deepwalk_emb))

    dw_emb = []
    for i in range(0, len(deepwalk_emb)):
        dw_emb.append(list(deepwalk_emb[i]))
    #   print(len(dw_emb))
    dw_emb = np.array(dw_emb)
    print(dw_emb)

    tsne = TSNE(n_components=2, verbose=1, random_state=0)
    z = tsne.fit_transform(dw_emb)
    z_data = np.vstack((z.T, labels)).T
    df_tsne = pd.DataFrame(z_data, columns=["Dim1", "Dim2", "class"])
    df_tsne["class"] = df_tsne["class"].astype(int)
    plt.figure(figsize=(8, 8))
    sns.scatterplot(
        data=df_tsne,
        hue="class",
        x="Dim1",
        y="Dim2",
        palette=["green", "orange", "brown", "red", "blue", "black"],
    )
    plt.savefig(
        "figs/dw_citeseer.pdf", bbox_inches="tight"
    )  # save embeddings if needed
    plt.savefig("figs/dw_citeseer.png", bbox_inches="tight")
    plt.show()

    print("Graph embedding via Node2Vec..............")
    node2vec_emb, _ = node2vec(
        g, dimensions=128, walk_length=80, num_walks=10, p=4, q=0.25
    )
    # print(node2vec_emb, len(node2vec_emb))

    n2v_emb = []
    for i in range(0, len(node2vec_emb)):
        n2v_emb.append(list(node2vec_emb[i]))
    # print(len(n2v_emb))
    n2v_emb = np.array(n2v_emb)
    print(n2v_emb)

    tsne = TSNE(n_components=2, verbose=1, random_state=0)
    z = tsne.fit_transform(n2v_emb)
    z_data = np.vstack((z.T, labels)).T
    df_tsne = pd.DataFrame(z_data, columns=["Dim1", "Dim2", "class"])
    df_tsne["class"] = df_tsne["class"].astype(int)
    plt.figure(figsize=(8, 8))
    sns.scatterplot(
        data=df_tsne,
        hue="class",
        x="Dim1",
        y="Dim2",
        palette=["green", "orange", "brown", "red", "blue", "black"],
    )

    plt.savefig("figs/n2v_citeseer.pdf", bbox_inches="tight")
    plt.savefig("figs/n2v_citeseer.png", bbox_inches="tight")
    plt.show()

    print("Graph embedding via LINE........")

    model = LINE(
        dimension=128,
        walk_length=80,
        walk_num=10,
        negative=5,
        batch_size=128,
        init_alpha=0.025,
        order=2,
    )

    model.train()
    line_emb = model(g, return_dict=True)

    l_emb = []
    for i in range(0, len(line_emb)):
        l_emb.append(list(line_emb[i]))
    #   print(len(l_emb))
    l_emb = np.array(l_emb)
    print(l_emb)

    tsne = TSNE(n_components=2, verbose=1, random_state=0)
    z = tsne.fit_transform(l_emb)
    z_data = np.vstack((z.T, labels)).T
    df_tsne = pd.DataFrame(z_data, columns=["Dim1", "Dim2", "class"])
    df_tsne["class"] = df_tsne["class"].astype(int)
    plt.figure(figsize=(8, 8))
    sns.scatterplot(
        data=df_tsne,
        hue="class",
        x="Dim1",
        y="Dim2",
        palette=["green", "orange", "brown", "red", "blue", "black"],
    )

    plt.savefig("figs/line_citeseer.pdf", bbox_inches="tight")
    plt.savefig("figs/line_citeseer.png", bbox_inches="tight")
    plt.show()

    print("Graph embedding via SDNE...........")
    model = eg.SDNE(
        g,
        node_size=len(g.nodes),
        nhid0=256,
        nhid1=32,
        dropout=0.025,
        alpha=5e-4,
        beta=10,
    )
    sdne_emb = model.train(model)

    sd_emb = []
    for i in range(0, len(sdne_emb)):
        sd_emb.append(list(sdne_emb[i]))
    #   print(len(sd_emb))
    sd_emb = np.array(sd_emb)
    print(sd_emb)

    tsne = TSNE(n_components=2, verbose=1, random_state=0)
    z = tsne.fit_transform(sd_emb)
    z_data = np.vstack((z.T, labels)).T
    df_tsne = pd.DataFrame(z_data, columns=["Dim1", "Dim2", "class"])
    df_tsne["class"] = df_tsne["class"].astype(int)
    plt.figure(figsize=(8, 8))
    sns.scatterplot(
        data=df_tsne,
        hue="class",
        x="Dim1",
        y="Dim2",
        palette=["green", "orange", "brown", "red", "blue", "black"],
    )

    plt.savefig("figs/sdne_citeseer2.pdf", bbox_inches="tight")
    plt.savefig("figs/sdne_citeseer2.png", bbox_inches="tight")
    plt.show()

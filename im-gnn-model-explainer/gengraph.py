"""gengraph.py

   Generating and manipulaton the synthetic graphs needed for the paper's experiments.
"""

import os

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.colors as colors

# Set matplotlib backend to file writing
plt.switch_backend("agg")

import networkx as nx

import numpy as np

from tensorboardX import SummaryWriter

from utils import synthetic_structsim,text_structure
from utils import featgen
import utils.io_utils as io_utils


####################################
#
# Experiment utilities
#
####################################
def perturb(graph_list, p):
    """ Perturb the list of (sparse) graphs by adding/removing edges.
    Args:
        p: proportion of added edges based on current number of edges.
    Returns:
        A list of graphs that are perturbed from the original graphs.
    """
    perturbed_graph_list = []
    for G_original in graph_list:
        G = G_original.copy()
        edge_count = int(G.number_of_edges() * p)
        # randomly add the edges between a pair of nodes without an edge.
        for _ in range(edge_count):
            while True:
                u = np.random.randint(0, G.number_of_nodes())
                v = np.random.randint(0, G.number_of_nodes())
                if (not G.has_edge(u, v)) and (u != v):
                    break
            G.add_edge(u, v)
        perturbed_graph_list.append(G)
    return perturbed_graph_list


def join_graph(G1, G2, n_pert_edges):
    """ Join two graphs along matching nodes, then perturb the resulting graph.
    Args:
        G1, G2: Networkx graphs to be joined.
        n_pert_edges: number of perturbed edges.
    Returns:
        A new graph, result of merging and perturbing G1 and G2.
    """
    assert n_pert_edges > 0
    F = nx.compose(G1, G2)
    edge_cnt = 0
    while edge_cnt < n_pert_edges:
        node_1 = np.random.choice(G1.nodes())
        node_2 = np.random.choice(G2.nodes())
        F.add_edge(node_1, node_2)
        edge_cnt += 1
    return F


def preprocess_input_graph(G, labels, normalize_adj=False):
    """ Load an existing graph to be converted for the experiments.
    Args:
        G: Networkx graph to be loaded.
        labels: Associated node labels.
        normalize_adj: Should the method return a normalized adjacency matrix.
    Returns:
        A dictionary containing adjacency, node features and labels
    """
    adj = np.array(nx.to_numpy_matrix(G))
    if normalize_adj:
        sqrt_deg = np.diag(1.0 / np.sqrt(np.sum(adj, axis=0, dtype=float).squeeze()))
        adj = np.matmul(np.matmul(sqrt_deg, adj), sqrt_deg)

    existing_node = list(G.nodes)[-1]
    feat_dim = G.nodes[existing_node]["feat"].shape[0]
    f = np.zeros((G.number_of_nodes(), feat_dim), dtype=float)
    for i, u in enumerate(G.nodes()):
        f[i, :] = G.nodes[u]["feat"]

    # add batch dim
    adj = np.expand_dims(adj, axis=0)
    f = np.expand_dims(f, axis=0)
    labels = np.expand_dims(labels, axis=0)
    return {"adj": adj, "feat": f, "labels": labels}



def gen_im(args):
    """ manually graph（networkx）
    https://blog.csdn.net/qq_23889009/article/details/102484550?ops_request_misc=&request_id=&biz_id=102&utm_term=networkx&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-0-102484550.nonecase&spm=1018.2226.3001.4187
    feature_label:feature array,label int;  adj:HeadNode EdgeType TailNode
    """
    basis_type = 'im'

    feature_label_path = os.path.join(args.datadir,args.graph_type, 'feature_label.csv')
    features_labels = pd.read_csv(feature_label_path,sep='\t')
    features = features_labels['feature'].values.tolist()
    labels = features_labels['label'].values.tolist()
    label2idx = {-1:0}
    idx2label = {0:-1}
    labels_set = set(labels)
    labels_set.remove(-1)
    for i,l in enumerate(labels_set):
        label2idx[l]=i+1
        idx2label[i+1]=l
    with open(os.path.join(args.datadir, args.graph_type,'label_map.txt'),'w',encoding='utf-8') as fw:
        fw.write(str(idx2label))
    labels = [label2idx[label] for label in labels]

    adj_path = os.path.join(args.datadir, args.graph_type, 'adj.txt')
    adj = []
    with open(adj_path,'r',encoding='utf-8') as f_adj:
        lines = f_adj.readlines()
        for line in lines[1:]:
            ni,_,nj = line.strip('\n').split('\t')
            adj.append([int(ni),int(nj)])

    plt.figure(figsize=(8, 6), dpi=len(labels))

    G, role_id = text_structure.build_graph(
        features, labels, adj
    )

    name = basis_type + "_" + str(len(features))
    return G, role_id, name
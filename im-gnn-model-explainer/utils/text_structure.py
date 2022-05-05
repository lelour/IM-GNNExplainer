#!/usr/bin/env python
# encoding: utf-8
"""
author: ql.hua 
create_dt: 2022/4/13 14:23
"""
import math

import networkx as nx
import numpy as np


def build_graph(
    features,
    labels,
    adj
):
    """This function creates a basis
    Possibility to add random edges afterwards.
    INPUT:
    --------------------------------------------------------------------------------------
    width_basis      :      width (in terms of number of nodes) of the basis
    add_random_edges :      nb of edges to randomly add on the structure
    OUTPUT:
    --------------------------------------------------------------------------------------
    basis            :      a nx graph with the particular shape
    role_ids         :      labels for each role
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(len(features)))
    feat_dict = {i: {'feat': np.array(eval(features[i]),dtype=np.float32)} for i in graph.nodes()}
    print('feat_dict[0]["feat"]:', feat_dict[0]['feat'].dtype)
    nx.set_node_attributes(graph, feat_dict)
    print('G.nodes[0]["feat"]:', graph.nodes[0]['feat'].dtype)
    for edge in adj:
        graph.add_edges_from([(edge[0], edge[1])])


    return graph, labels
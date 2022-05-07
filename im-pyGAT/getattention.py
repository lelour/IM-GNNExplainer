#!/usr/bin/env python
# encoding: utf-8
"""
author: Jian Li 
create_dt: 2022/4/25 20:37
"""

from __future__ import division
from __future__ import print_function
from src import io_utils

import os
import glob
import time
import random
import argparse
import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

from models import GAT

plt.rcParams['font.family']='SimHei'
plt.rcParams['axes.unicode_minus']=False

def show_adjacency_full(nodeid_att, node_idx_new, ax=None):
    adj = np.zeros((len(nodeid_att), len(nodeid_att)))
    for i in range(len(nodeid_att)):
        if i != node_idx_new:
            adj[i, node_idx_new] = nodeid_att[i]
            adj[node_idx_new][i] = nodeid_att[i]
    if ax is None:
        plt.figure()
        plt.imshow(adj)
        # nodeid_att_nei = nodeid_att[torch.arange(nodeid_att.size(0)) != node_idx_new]
        # plt.imshow(np.array(nodeid_att_nei.detach().numpy()).T, cmap=plt.get_cmap("BuPu"))
        # cbar = plt.colorbar(orientation='horizontal')
        # cbar.solids.set_edgecolor("face")
    else:
        ax.imshow(adj)


    return adj

def filter_adj(adj):
    filt_adj = adj.copy()
    filt_adj[adj<0.8] = 0
    return filt_adj


def get_nodecolor(neighbors, nodes, nodeidx, pred_tgt):
    ''' 根据节点GCN分类pred，给节点不同的颜色:nodeidx为红色，99-灰色，0-蓝色，1-黄色，2-绿色，4-橙色，'''
    color_map = {'未知标签':'silver', '非投诉会话':'lightsteelblue', '客户感受差':'gold', '投诉/曝光':'yellowgreen', '未解决问题':'lightsalmon', '四到类':'violet','词':'thistle'}

    neighbors_gcnpred = [color_map['词'] if neighbors[n] != nodeidx else color_map[pred_tgt] for n in nodes]  # if int(nei)!= int(nodeidx) else 'red'
    return neighbors_gcnpred

def get_neighbors_oriinfo(neighbors,nodes,nodeid2txt,label_map_pred,pred,labels):
    ''' 获取neighbors 的中文 '''
    if len(neighbors) and len(nodes):
        neighbors_ = [neighbors[node] for node in nodes]
        neighbors_txts_labels = [[nodeid2txt[nei],label_map_pred[int(labels[nei])],label_map_pred[int(np.argmax(pred[nei]))]] for nei in neighbors_]
        return neighbors_txts_labels
    return []

def paint(nodeid_att, nodeid, node_idx_new, neibor,pred_tgt,nodeid2txt,label_map_pred,pred,labels):

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # Full adjacency
    ax1.set_title('Full Adjacency mask of Subgraph')
    adj = show_adjacency_full(nodeid_att,node_idx_new, ax=ax1)

    # Filtered adjacency
    # f = filter_adj(adj)
    # ax2.set_title('Full Adjacency mask of Subgraph')
    # ax2.imshow(f);

    # Plot subgraph
    ax2.set_title(f"Subgraph of '{nodeid2txt[nodeid]}'")
    G = nx.from_numpy_array(adj)
    G.remove_nodes_from(list(nx.isolates(G)))
    node_colors = get_nodecolor(neibor, G.nodes, nodeid, pred_tgt)
    nx.draw(G, pos=nx.circular_layout(G), with_labels=True, ax=ax2, node_color=node_colors)

    # Plot nodeidx2txt
    ax3.set_title(f"Mapping of nodeidx and text")
    neighbors_txts_labels = get_neighbors_oriinfo(neibor, G.nodes,nodeid2txt,label_map_pred,pred,labels)
    if len(neighbors_txts_labels):
        tab = plt.table(cellText=neighbors_txts_labels,
                        colLabels=['txt', 'label','pred'],
                        rowLabels=[str(n) for n in G.nodes],
                        loc='center',
                        cellLoc='center',
                        rowLoc='center')
        tab.auto_set_font_size(True)
        tab.set_fontsize(16)
        tab.scale(1, 2)
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', dest='logdir',default='log', help='Tensorboard log directory')
    parser.add_argument('--ckptdir', dest='ckptdir',default='model_doc_word_sen', help='Tensorboard log directory')
    parser.add_argument('--dataset', dest='dataset', default='im', help='Input dataset.')
    parser.add_argument("--graph_type", dest="graph_type",default='doc_word_sen', help="context_centernode or doc_word")
    parser.add_argument("--label_idx", dest="label_idx",default='5四到类', help="label_idx")
    parser.add_argument('--method', dest='method', default='GAT', help='Method. Possible values: base, ')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
    parser.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
    parser.add_argument('--seed', type=int, default=72, help='Random seed.')
    parser.add_argument('--train_ratio', dest='train_ratio', type=float, default=0.8, help='Ratio of number of graphs training set to all graphs.')
    parser.add_argument('--val_ratio', dest='val_ratio', type=float, default=0.2, help='Ratio of number of graphs training set to all graphs.')
    parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.')
    parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
    parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
    parser.add_argument('--patience', type=int, default=100, help='Patience')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    ckpt = io_utils.load_ckpt(args)
    cg_dict = ckpt["cg"] # get computation graph
    adj,feats,labels,pred = cg_dict['adj'],cg_dict['feat'],cg_dict['label'],cg_dict['pred']
    input_dim = cg_dict["feat"].shape[1]
    num_classes = cg_dict["pred"].shape[1]
    print("Loaded model from {}".format(args.ckptdir))
    print("input dim: ", input_dim, "; num classes: ", num_classes)

    features, adj, labels = Variable(feats), Variable(adj), Variable(labels)

    nodeid2txt = {}
    path = 'data'
    nodeid2txt_path = os.path.join(path,args.graph_type,'nodeid2txt.txt')
    with open(nodeid2txt_path,'r',encoding='utf-8') as fr:
        lines = fr.readlines()
        for line in lines[1:]:
            nodeidx,txt = line.strip('\n').split('\t')
            nodeid2txt[int(nodeidx)]=txt
    label_map_pred = {0: '未知标签', 1: '非投诉会话', 2:'客户感受差', 3: '投诉/曝光', 4: '未解决问题', 5:'四到类',6:'词'}

    node_indices = io_utils.get_nodeidx(args)

    model = GAT(nfeat=feats.shape[1],
                nhid=args.hidden,
                nclass=7,
                dropout=args.dropout,
                nheads=args.nb_heads,
                alpha=args.alpha)
    model.load_state_dict(torch.load(args.ckptdir+'/{}.pkl'.format(68)))
    att_final = model.get_final_att(features, adj)

    fw_label_idx = open(r"gnn-model-explainer-master\log\subgraph\gat"+"\\"+args.label_idx+"_1.txt",'w',encoding='utf-8')

    print(len(node_indices))
    acc = 0
    for nodeid in node_indices:
        txt_tgt = nodeid2txt[nodeid]
        pred_tgt = label_map_pred[np.argmax(pred[nodeid])]
        label_tgt = label_map_pred[int(labels[nodeid].cpu().detach().numpy())]
        acc += bool(np.argmax(pred[nodeid])==int(labels[nodeid].cpu().detach().numpy()))
        node_idx_new, sub_adj, sub_feat, sub_label, neighbors = io_utils.extract_neighborhood(adj,feats,labels,nodeid)
        nodeid_att = att_final[nodeid, neighbors]
        nodeid_att_indexes = np.where(nodeid_att > 0)
        txt_neis = []
        for i in nodeid_att_indexes[0]:
            txt_nei = nodeid2txt[neighbors[i]]
            txt_neis.append(txt_nei)
        fw_label_idx.write('\t'.join([txt_tgt,pred_tgt,label_tgt,str(txt_neis)])+'\n')

        ## 画图 所有attention
        paint(nodeid_att, nodeid, node_idx_new, neighbors,pred_tgt,nodeid2txt,label_map_pred,pred,labels)
    fw_label_idx.close()
    # print(acc,len(node_indices),acc/len(node_indices))


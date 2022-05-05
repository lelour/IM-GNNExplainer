#!/usr/bin/env python
# encoding: utf-8
"""
author: ql.hua 
create_dt: 2022/4/25 16:17
"""
import os
import torch
import numpy as np


def gen_prefix(args):
    '''Generate label prefix for a graph model.
    '''
    # if args.bmname is not None:
    #     name = args.bmname
    # else:
    #     name = args.dataset
    name = args.dataset
    name += "_" + args.graph_type + "_" + args.method
    name += "_h" + str(args.hidden)
    return name

def create_filename(save_dir, args, isbest=False, num_epochs=-1):
    """
    Args:
        args        :  the arguments parsed in the parser
        isbest      :  whether the saved model is the best-performing one
        num_epochs  :  epoch number of the model (when isbest=False)
    """
    filename = os.path.join(save_dir, gen_prefix(args))
    os.makedirs(filename, exist_ok=True)

    if isbest:
        filename = os.path.join(filename, "best")
    elif num_epochs > 0:
        filename = os.path.join(filename, str(num_epochs))

    return filename + ".pth.tar"

def save_checkpoint(model, optimizer, args, num_epochs=-1, isbest=False, cg_dict=None):
    """Save pytorch model checkpoint.

    Args:
        - model         : The PyTorch model to save.
        - optimizer     : The optimizer used to train the model.
        - args          : A dict of meta-data about the model.
        - num_epochs    : Number of training epochs.
        - isbest        : True if the model has the highest accuracy so far.
        - cg_dict       : A dictionary of the sampled computation graphs.
    """
    filename = create_filename(args.ckptdir, args, isbest, num_epochs=num_epochs)
    print(f"filename: {filename}")
    torch.save(
        {
            "epoch": num_epochs,
            "model_type": args.method,
            "optimizer": optimizer,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "cg": cg_dict,
        },
        filename,
    )


def load_ckpt(args, isbest=False):
    '''Load a pre-trained pytorch model from checkpoint.
    '''
    print("loading model")
    filename = create_filename(args.ckptdir, args, isbest)
    print(filename)
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        ckpt = torch.load(filename)
    else:
        print("Checkpoint does not exist!")
        print("Checked path -- {}".format(filename))
        print("Make sure you have provided the correct path!")
        print("You may have forgotten to train a model for this dataset.")
        print()
        print("To train one of the paper's models, run the following")
        print(">> python train.py --dataset=DATASET_NAME")
        print()
        raise Exception("File not found.")
    return ckpt


def get_nodeidx(args):
    """ 获取subgraph的 nodeidx"""
    node_idx_list = []
    files = os.listdir(os.path.join(r"E:\0_code\gnn-model-explainer-master_hql\log\subgraph\exp\doc_word_sen\all_subgraph", args.label_idx))
    for file in files:
        file_params = file.split('_')
        for i in file_params:
            if 'graph' in i:
                graphidx = i[:-5]
        node_idx_list.append(int(graphidx))
    return node_idx_list

def get_allneighborhoods(adj, n_hops=3, use_cuda=False):
    """Returns the n_hops degree adjacency matrix adj."""
    adj = torch.tensor(adj, dtype=torch.float)
    if use_cuda:
        adj = adj.cuda()
    hop_adj = power_adj = adj
    for i in range(n_hops - 1):
        power_adj = power_adj @ adj
        prev_hop_adj = hop_adj
        hop_adj = hop_adj + power_adj
        hop_adj = (hop_adj > 0).float()
    return hop_adj.cpu().numpy().astype(int)

def extract_neighborhood(adj,feat,label,node_idx):
    """Returns the neighborhood of a given node."""
    neighborhoods = get_allneighborhoods(adj)
    neighbors_adj_row = neighborhoods[node_idx, :]
    # index of the query node in the new adj
    node_idx_new = sum(neighbors_adj_row[:node_idx])
    neighbors = np.nonzero(neighbors_adj_row)[0]
    sub_adj = adj[neighbors][:, neighbors] # neighbors 内部连接情况
    sub_feat = feat[neighbors]
    sub_label = label[neighbors]
    return node_idx_new, sub_adj, sub_feat, sub_label, neighbors

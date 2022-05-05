import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch


def encode_onehot(labels):
    # The classes must be sorted before encoding to enable static class encoding.
    # In other words, make sure the first class always maps to index 0.
    classes = sorted(list(set(labels)))
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot,classes


def load_data(path="./data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize_features(features)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    adj = torch.FloatTensor(np.array(adj.todense()))
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def load_data(args,logpath):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(' '.join([args.dataset,args.graph_type])))

    feature_label_path = os.path.join('data', args.dataset, args.graph_type,'feature_label.csv')
    features_labels = pd.read_csv(feature_label_path, sep='\t')
    features = []
    for f in features_labels['feature'].values.tolist():
        features.append(eval(f))
    features = np.array(features,dtype=np.float32)
    labels = np.array(features_labels['label'].values.tolist())
    features = sp.csr_matrix(features, dtype=np.float32)
    labels, classes = encode_onehot(labels)
    with open(os.path.join('data', args.dataset, args.graph_type,'label_map.txt'),'w',encoding='utf-8') as fw:
        fw.write(str({i: c for i, c in enumerate(classes)}))

    # build graph adj
    idx = np.array(features_labels['nodeid'].values, dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}

    adj_path = os.path.join('data', args.dataset, args.graph_type, 'adj.txt')
    edges_unordered = []
    with open(adj_path, 'r', encoding='utf-8') as f_adj:
        lines = f_adj.readlines()
        for line in lines[1:]:
            ni, _, nj = line.strip('\n').split('\t')
            edges_unordered.append([int(ni), int(nj)])

    edges_unordered = np.array(edges_unordered, dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize_features(features)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    idx = features_labels['nodeid'].values.tolist()
    np.random.shuffle(idx)
    num_train = int(len(idx) * args.train_ratio)
    num_val = num_train + int(len(idx)* args.val_ratio)
    idx_train = idx[:num_train]
    idx_val = idx[num_train:]
    # idx_test = idx[num_val:]

    np.save(os.path.join(logpath,'train_idx.npy'),np.array(idx_train))
    np.save(os.path.join(logpath,'val_idx.npy'),np.array(idx_val))
    # np.save(os.path.join(logpath,'test_idx.npy'),np.array(idx_test))

    adj = torch.FloatTensor(np.array(adj.todense()))
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


''' https://github.com/Diego999/pyGAT'''
from __future__ import division
from __future__ import print_function

import os
import glob
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import sklearn.metrics as metrics
from tensorboardX import SummaryWriter

from utils import load_data, accuracy
from models import GAT, SpGAT
from src.io_utils import gen_prefix,save_checkpoint

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--logdir', dest='logdir',default='log', help='Tensorboard log directory')
parser.add_argument('--ckptdir', dest='ckptdir',default='model_doc_word_sen', help='Tensorboard log directory')
parser.add_argument('--dataset', dest='dataset', default='im', help='Input dataset.')
parser.add_argument("--graph_type", dest="graph_type",default='doc_word_sen', help="context_centernode or doc_word")
parser.add_argument('--method', dest='method', default='GAT', help='Method. Possible values: base, ')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--train_ratio', dest='train_ratio', type=float, default=0.8, help='Ratio of number of graphs training set to all graphs.')
parser.add_argument('--val_ratio', dest='val_ratio', type=float, default=0.2, help='Ratio of number of graphs training set to all graphs.')
parser.add_argument('--epochs', type=int, default=2, help='Number of epochs to train.')
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

path = os.path.join(args.logdir, gen_prefix(args))
logpath = path+ '-' + time.strftime('%m_%d_%H_%M', time.localtime())
# writer = SummaryWriter(logpath)

# Load data
adj, features, labels, idx_train, idx_val = load_data(args,logpath)

# Model and optimizer
if args.sparse:
    model = SpGAT(nfeat=features.shape[1], 
                nhid=args.hidden, 
                nclass=int(labels.max())+1,
                dropout=args.dropout, 
                nheads=args.nb_heads, 
                alpha=args.alpha)
else:
    model = GAT(nfeat=features.shape[1], 
                nhid=args.hidden, 
                nclass=int(labels.max())+1,
                dropout=args.dropout, 
                nheads=args.nb_heads, 
                alpha=args.alpha)
optimizer = optim.Adam(model.parameters(), 
                       lr=args.lr, 
                       weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()

features, adj, labels = Variable(features), Variable(adj), Variable(labels)

def evaluate_node(ypred, labels, train_idx, test_idx):
    _, pred_labels = torch.max(ypred, 1)
    pred_labels = pred_labels.numpy()

    pred_train = np.ravel(pred_labels[train_idx])
    pred_test = np.ravel(pred_labels[test_idx])
    labels_train = np.ravel(labels[train_idx])
    labels_test = np.ravel(labels[test_idx])

    result_train = {
        "prec": metrics.precision_score(labels_train, pred_train, average="macro"),
        "recall": metrics.recall_score(labels_train, pred_train, average="macro"),
        "acc": metrics.accuracy_score(labels_train, pred_train),
        "conf_mat": metrics.confusion_matrix(labels_train, pred_train),
    }
    result_test = {
        "prec": metrics.precision_score(labels_test, pred_test, average="macro"),
        "recall": metrics.recall_score(labels_test, pred_test, average="macro"),
        "acc": metrics.accuracy_score(labels_test, pred_test),
        "conf_mat": metrics.confusion_matrix(labels_test, pred_test),
    }
    return result_train, result_test

def train(epoch, writer):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    result_train, result_test = evaluate_node(
        output.cpu(), labels, idx_train, idx_val
    )
    if writer is not None:
        writer.add_scalars(
            "loss",
            {"train":loss_train, "test": loss_val},
            epoch)
        writer.add_scalars(
            "prec",
            {"train": result_train["prec"], "test": result_test["prec"]},
            epoch,
        )
        writer.add_scalars(
            "recall",
            {"train": result_train["recall"], "test": result_test["recall"]},
            epoch,
        )
        writer.add_scalars(
            "acc", {"train": result_train["acc"], "test": result_test["acc"]}, epoch
        )
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.data.item()),
          'acc_train: {:.4f}'.format(result_train['acc']),
          'loss_val: {:.4f}'.format(loss_val.data.item()),
          'acc_val: {:.4f}'.format(result_test['acc']),
          'time: {:.4f}s'.format(time.time() - t))

    return loss_val.data.item()


def compute_test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output, labels)
    acc_test = accuracy(output, labels)
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.data.item()),
          "accuracy= {:.4f}".format(acc_test.data.item()))
    return output

# Train model
t_total = time.time()
loss_values = []
bad_counter = 0
best = args.epochs + 1
best_epoch = 0
for epoch in range(args.epochs):
    loss_values.append(train(epoch,writer))

    torch.save(model.state_dict(), args.ckptdir+'/{}.pkl'.format(epoch))
    if loss_values[-1] < best:
        best = loss_values[-1]
        best_epoch = epoch
        bad_counter = 0
    else:
        bad_counter += 1

    if bad_counter == args.patience:
        break

    files = glob.glob(args.ckptdir+'/*.pkl')
    for file in files:
        epoch_nb = int(file.split('\\')[1].split('.')[0])
        if epoch_nb < best_epoch:
            os.remove(file)

files = glob.glob(args.ckptdir+'/*.pkl')
for file in files:
    epoch_nb = int(file.split('\\')[1].split('.')[0])
    if epoch_nb > best_epoch:
        os.remove(file)

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Restore best model
print('Loading {}th epoch'.format(best_epoch))
model.load_state_dict(torch.load(args.ckptdir+'/{}.pkl'.format(best_epoch)))

# Testing
output = compute_test()
cg_data = {
        "adj": adj,
        "feat": features,
        "label": labels,
        "pred": output.cpu().detach().numpy(),
        "train_idx": idx_train,
        "val_idx": idx_val
    }
save_checkpoint(model, optimizer, args, num_epochs=best_epoch, isbest=True, cg_dict=cg_data)

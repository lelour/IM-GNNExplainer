# Pytorch Graph Attention Network

# Requirements

pyGAT relies on Python 3.5 and PyTorch 0.4.1 (due to torch.sparse_coo_tensor).

## Projector Directory Introduction
| Folder Name     | Description  | Internal Structure Description|
|----------|-------------------|--------------|
| data  | store related datasets of train.py  |File naming rules：DATASET/'GRAPH_TYPE'|
| log  | store all log of experiments.  |**npy**:store explanation results of different methods(exp,att,grad,gat)（**values**）<br>npy directory structure：method/graph_type/label_idx/*.npy files<br><br>**subgraph**:Store explanation results（**subgraph**）<br>subgraph directory structure：同npy<br><br>**other files**：train.py generated model training progress tfevents|
| model_doc_word_sen  | store pre-trained GAT model of train.py  |File naming rules：<br>'EXPERIMENT_NAME'_'GRAPH_TYPE'_'method'_h'dimension of GCN hidden layer'.pth.tar and 68.pkl They are different saving forms of the same model|
| src  | Store additional script files  |io_utils.py Script for configuring io|
| getattention.py  | Gets the value of the last layer of attention <br>Calculate and save the sub graph node and draw the sub graph <br>**Different from ATT in Explainer**  |/|
|layers.py |model layer script|/|
|models.py |model script file|/|
|train.py |GAT training script|/|

# Pytorch Graph Attention Network

# Performances

# Requirements

pyGAT relies on Python 3.5 and PyTorch 0.4.1 (due to torch.sparse_coo_tensor).

# Issues/Pull Requests/Feedbacks

Don't hesitate to contact for any feedback or create issues/pull requests.

## Projector 目录介绍
| 文件夹名称     | Description  | 内部结构介绍|
|----------|-------------------|--------------|
| data  | 存储train.py涉及的数据集  |文件命名规则：DATASET/'GRAPH_TYPE'|
| log  | 实验所有log存放.  |**npy**:存放不同method(exp,att,grad,gat)的解释结果（**数值**）<br>npy目录结构：method/graph_type/label_idx/*.npy文件<br><br>**subgraph**:存放解释结果（**子图**）<br>subgraph目录结构：同npy<br><br>**其他文件**：train.py生成的模型训练tfevents|
| model_doc_word_sen  | 存储train.py训练好的GAT模型  |文件命名规则：<br>'EXPERIMENT_NAME'_'GRAPH_TYPE'_'method'_h'GCN隐藏层维数'.pth.tar 和 68.pkl是同一个模型的不同保存形式|
| src  | 存储其他脚本文件  |io_utils.py 配置io的脚本|
| getattention.py  | 获取最后一层attention的值 <br>计算保存子图节点，并画出子图 <br>**不同于explainer中的att**  |/|
|layers.py |模型层脚本|/|
|models.py |模型脚本文件|/|
|train.py |GAT训练脚本|/|

# im-gnn-explainer

## Using the explainer

#### Training a GCN model 

This is the model that will be explained. To re-train these models, run the following:

##### My Dataset：EXPERIMENT_NAME = im
#####Graph Construction： 

| Name     | `GRAPH_TYPE` | Description  | 节点特征  | 边关系构建  |
|----------|-------------------|--------------|--------------|--------------|
| Context  | `context_centernode`  | 相似句子聚合成中心节点和剩余节点 <br><br>中心节点选取：最接近所属类点特征的平均值的节点|Bert-Whitening output | [句子](#)在session中是紧邻的上下句则有边   |
| Words&Sentence (Balanced) | `doc_word_sen`  | 句子为单独的个体，节点为句子清洗（分词去停）后的词库和句子  | 词节点：one-hot<br>句节点：BOW|[句词](#)：包含关系 （句子A中有词B则AB有边） <br>[词词](#)：共现关系（词B和词C曾同时出现在一个句子中则有边）|
| 词句+上下文构图 | `context_doc_word_sen`  | 在词句构图基础上追加了句间的边，打乱后随机取8：2=train:test 划分数据集| 词节点：one-hot<br>句节点：BOW|[句词](#)：包含关系 （句子A中有词B则AB有边） <br>[词词](#)：共现关系 <br> [句句](#): 句子在session中紧邻的上下句则有边|
| 词句+上下文构图，按标签分布划分数据集 | `context_doc_word_sen_noshuffle`  | 较于context_doc_word_sen，区别在于划分数据集方式:按标签分布平均分配在trainset和testset中| 词节点：one-hot<br>句节点：BOW|[句词](#)：包含关系 （句子A中有词B则AB有边） <br>[词词](#)：共现关系 <br> [句句](#): 句子在session中紧邻的上下句则有边|

```
python train.py --dataset=EXPERIMENT_NAME --graph_type=GRAPH_TYPE
```

where `EXPERIMENT_NAME` is the experiment you want to replicate. 

For a complete list of options in training the GCN models:

```
python train.py --help
```

> TODO: Explain outputs

#### Explaining a GCN model

To run the explainer, run the following:

```
python explainer_main.py --dataset=EXPERIMENT_NAME --graph_type=GRAPH_TYPE --explain_nodes=EXPLAIN_NODES
```

where `EXPERIMENT_NAME` is the experiment you want to replicate.

| Name      |`LABEL_IDX`|`EXPLAIN_NODES`(Words&Sentence graph construction)| 
|----------|-------------------|-----------------|
|未知标签|-1未知标签|range(786,836)|
|非投诉会话|0非投诉会话|range(536,586)|
|客户感受差|1客户感受差|range(586,635)|
|投诉会话|2投诉会话|range(636,686)|
|未解决问题|4未解决问题|range(686,736)|
|四到类|5四到类|range(736,786)|


For a complete list of options provided by the explainer:

```
python train.py --help
```

#### Visualizing the explanations

##### Tensorboard

The result of the optimization can be visualized through Tensorboard.

**LOGDIR**：events.out.tfevents.* Folder level directory of the file
<br>e.g. .\im_doc_word_sen_att_h512_o256-04_25_18_04\
```
tensorboard --logdir LOGDIR
```

You should then have access to visualizations served from `localhost`.

#### Jupyter Notebook

We provide an example visualization through Jupyter Notebooks in the `notebook` folder. To try it:

```
jupyter notebook
```

The default visualizations are provided in `notebook/GNN-Explainer-Viz.ipynb`.

> Note: For an interactive version, you must enable ipywidgets
>
> ```
> jupyter nbextension enable --py widgetsnbextension
> ```

You can now play around with the mask threshold in the `GNN-Explainer-Viz-interactive.ipynb`.
> TODO: Explain outputs + visualizations + baselines

#### D3,js

We provide export functionality so the generated masks can be visualized in other data visualization 
frameworks, for example [d3.js](http://observablehq.com). We provide [an example visualization in Observable](https://observablehq.com/d/00c5dc74f359e7a1).

#### Included experiments

| Name     | `EXPERIMENT_NAME` | Description  |
|----------|:-------------------:|--------------|
| IM | `im`  | 我们的IM会话客人文本<br>multiple graph construction methods  |

> Datasets with a * are passed with the `--bmname` parameter rather than `--dataset` as they require being downloaded manually.


### Using the explainer on other models
A graph attention model is provided. This repo is still being actively developed to support other
GNN models in the future.

## Changelog

## Projector Directory Introduction
| Folder Name     | Description  | Internel Structure Description|
|----------|-------------------|--------------|
| ckpt  | 存储train.py训练好的GCN模型  |File naming rules：<br>'EXPERIMENT_NAME'_'GRAPH_TYPE'_'method'_h'dimension of GCN hidden layers'_o'dimension of GCN output layer'.pth.tar|
| data  | 存储train.py涉及的数据集  |File naming rules：'GRAPH_TYPE'|
| explainer  | Implementation of the explainer.  |/|
| log  | Storage of all experimental logs.  |**npy**:存放不同method(exp,att,grad,gat)的解释结果（**数值**）<br>npy directory structure：method/graph_type/label_idx/*.npy files<br><br>**subgraph**:store explanation results（**subgraph**）<br>subgraph directory structure： same as npy<br><br>**other files**：train.py生成的模型训练tfevents|
| notebook  | GNN-Explainer - visualization execute file, Words&Sentence graph construction  |GNN-Explainer-Viz-att-doc_word_sen.ipynb : gnnexplainer integrated att<br>GNN-Explainer-Viz-grad-doc_word_sen.ipynb：gnnexplainer integrated grad<br>GNN-Explainer-Viz.ipynb：gnnexplainer method<br>remaining files：notebook|
| utils  | some common script files  |/|
| check_pred.py  | check prediction - output the GCN prediction overall result and sentence node result according to the model file<br>calculate and save the metrics of pred and label  |/|
|configs.py|train.py corresponding configuration file|/|
|explainer_main.py|gnnexplainer execute file|/|
|gengraph.py |graph construction script|/|
|models.py |model script file|/|
|train.py |GCN training script|/|

**NOTICE：** Modify the parameter setting position of different methods

explainer integrated [att](): In configs.py, parser.set_defaults(method='att')

explainer integrated [grad](): In explainer_main.py, masked_adj = explainer.explain_nodes_gnn_stats(range(655), prog_args,model='grad')

[explainer]()（exp）: **Both 1 and 2 are required**

    1.In configs.py, parser.set_defaults(method='base')

    2.In explainer_main.py, masked_adj = explainer.explain_nodes_gnn_stats(range(655), prog_args,model='exp')
[gat]():Open pyGAT Project





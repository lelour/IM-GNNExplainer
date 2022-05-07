# im-gnn-explainer

## Using the explainer

#### Training a GCN model 

This is the model that will be explained. To re-train these models, run the following:

##### My Dataset：EXPERIMENT_NAME = im
#####Graph Construction： 

| Name     | `GRAPH_TYPE` | Description  | Node Features  | Construct Edge Relationships  |
|----------|-------------------|--------------|--------------|--------------|
| Context  | `context_centernode`  | Similar sentences are aggregated into central nodes and rest nodes <br><br>Central node selection: the node closest to the average value of the characteristics of the class point|Bert-Whitening output | [Sentence](#)In a session, the next sentence that is immediately adjacent has an edge   |
| Words&Sentence (Balanced) | `doc_word_sen`  | The sentence is an individual, and the node is the thesaurus and sentence after sentence cleaning (word segmentation and remove stopwords)  | word node：one-hot<br>Sentence node：BOW|[word-sentence](#)：Inclusion relationship （If there is word B in sentence A, there is edge AB） <br>[word-word](#)：co-occurrence relationship（Words B and C have appeared in a sentence at the same time, then they have edges）|
| Words&Context | `context_doc_word_sen`  | The edges between sentences are added on the basis of word and sentence construction, which are taken randomly after being disturbed 8：2=train:test split dataset| Word Node：one-hot<br>Sentence Node：BOW|[word-sentence](#)：Inclusion Relationship （If there is word B in sentence A, there is edge AB） <br>[word-word](#)：co-occurrence relationship <br> [sentence-sentence](#): The next sentence in a session has an edge|
| Words&Context，Divide datasets by label distribution | `context_doc_word_sen_noshuffle`  | Compared with context_ doc_ word_ Sen, the difference lies in the way the dataset is divided:It is uniform distributed in trainset and testset according to label distribution| Word Node：one-hot<br>Sentence Node：BOW|[word-sentence](#)：Inclusion Relationship （If there is word B in sentence A, there is edge AB） <br>[word-word](#)：co-occurrence relationship <br> [sentence-sentence](#): The next sentence in a session has an edge|

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
|Unknown Label|-1 Unknown Label|range(786,836)|
|Non complaint session|0 Non complaint session|range(536,586)|
|Poor customer experience|1 Poor customer experience|range(586,635)|
|Complaint conversation|2 Complaint conversation|range(636,686)|
|Unresolved issues|4 Unresolved issues|range(686,736)|
|Arrived|5 Arrived|range(736,786)|


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
| IM | `im`  | Our IM conversation guest text<br>multiple graph construction methods  |

> Datasets with a * are passed with the `--bmname` parameter rather than `--dataset` as they require being downloaded manually.


### Using the explainer on other models
A graph attention model is provided. This repo is still being actively developed to support other
GNN models in the future.

## Changelog

## Projector Directory Introduction
| Folder Name     | Description  | Internel Structure Description|
|----------|-------------------|--------------|
| ckpt  |  Store pre-trained GCN model by train.py  |File naming rules：<br>'EXPERIMENT_NAME'_'GRAPH_TYPE'_'method'_h'dimension of GCN hidden layers'_o'dimension of GCN output layer'.pth.tar|
| data  | Store related datasets with train.py  |File naming rules：'GRAPH_TYPE'|
| explainer  | Implementation of the explainer.  |/|
| log  | Storage of all experimental logs.  |**npy**: Store the explanation results of different methods(exp,att,grad,gat)（**values**）<br>npy directory structure：method/graph_type/label_idx/*.npy files<br><br>**subgraph**:store explanation results（**subgraph**）<br>subgraph directory structure： same as npy<br><br>**other files**：train.py generated model training progress tfevents|
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





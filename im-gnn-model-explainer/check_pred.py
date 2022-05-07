#!/usr/bin/env python
# encoding: utf-8
"""
author: Jian Li 
create_dt: 2022/4/14 16:13
"""
import os
import argparse
import numpy as np
import pandas as pd
import utils.io_utils as io_utils
import utils.parser_utils as parser_utils
from sklearn.metrics import precision_recall_fscore_support,classification_report

def arg_parse():
    parser = argparse.ArgumentParser(description="GNN Explainer arguments.")
    # io_parser = parser.add_mutually_exclusive_group(required=False)
    # io_parser.add_argument("--dataset", dest="dataset", help="Input dataset.")
    # benchmark_parser = io_parser.add_argument_group()
    # benchmark_parser.add_argument(
    #     "--bmname", dest="bmname", help="Name of the benchmark dataset"
    # )
    # io_parser.add_argument("--pkl", dest="pkl_fname", help="Name of the pkl data file")

    parser_utils.parse_optimizer(parser)

    parser.add_argument("--clean-log", action="store_true", help="If true, cleans the specified log directory before running.")
    parser.add_argument("--logdir", dest="logdir", help="Tensorboard log directory")
    parser.add_argument("--ckptdir", dest="ckptdir", help="Model checkpoint directory")
    parser.add_argument('--datadir', dest='datadir',help='Directory where benchmark is located')
    parser.add_argument("--graph_type", dest="graph_type", help="context_centernode or doc_word")
    parser.add_argument("--cuda", dest="cuda", help="CUDA.")
    parser.add_argument(
        "--gpu",
        dest="gpu",
        action="store_const",
        const=True,
        default=False,
        help="whether to use GPU.",
    )
    parser.add_argument(
        "--epochs", dest="num_epochs", type=int, help="Number of epochs to train."
    )
    parser.add_argument(
        "--hidden-dim", dest="hidden_dim", type=int, help="Hidden dimension"
    )
    parser.add_argument(
        "--output-dim", dest="output_dim", type=int, help="Output dimension"
    )
    parser.add_argument(
        "--num-gc-layers",
        dest="num_gc_layers",
        type=int,
        help="Number of graph convolution layers before each pooling",
    )
    parser.add_argument(
        "--bn",
        dest="bn",
        action="store_const",
        const=True,
        default=False,
        help="Whether batch normalization is used",
    )
    parser.add_argument("--dropout", dest="dropout", type=float, help="Dropout rate.")
    parser.add_argument(
        "--nobias",
        dest="bias",
        action="store_const",
        const=False,
        default=True,
        help="Whether to add bias. Default to True.",
    )
    parser.add_argument(
        "--no-writer",
        dest="writer",
        action="store_const",
        const=False,
        default=True,
        help="Whether to add bias. Default to True.",
    )
    # Explainer
    parser.add_argument("--mask-act", dest="mask_act", type=str, help="sigmoid, ReLU.")
    parser.add_argument(
        "--mask-bias",
        dest="mask_bias",
        action="store_const",
        const=True,
        default=False,
        help="Whether to add bias. Default to True.",
    )
    parser.add_argument(
        "--explain-node", dest="explain_node", type=int, help="Node to explain."
    )
    parser.add_argument(
        "--graph-idx", dest="graph_idx", type=int, help="Graph to explain."
    )
    parser.add_argument(
        "--graph-mode",
        dest="graph_mode",
        action="store_const",
        const=True,
        default=False,
        help="whether to run Explainer on Graph Classification task.",
    )
    parser.add_argument(
        "--multigraph-class",
        dest="multigraph_class",
        type=int,
        help="whether to run Explainer on multiple Graphs from the Classification task for examples in the same class.",
    )
    parser.add_argument(
        "--multinode-class",
        dest="multinode_class",
        type=int,
        help="whether to run Explainer on multiple nodes from the Classification task for examples in the same class.",
    )
    parser.add_argument(
        "--align-steps",
        dest="align_steps",
        type=int,
        help="Number of iterations to find P, the alignment matrix.",
    )

    parser.add_argument(
        "--method", dest="method", type=str, help="Method. Possible values: base, att."
    )
    parser.add_argument(
        "--name-suffix", dest="name_suffix", help="suffix added to the output filename"
    )
    parser.add_argument(
        "--explainer-suffix",
        dest="explainer_suffix",
        help="suffix added to the explainer log",
    )

    # TODO: Check argument usage
    parser.set_defaults(
        logdir="log",
        ckptdir="ckpt",
        datadir="data",
        dataset="im",
        graph_type="doc_word_sen", #context_centernode
        opt="adam",
        opt_scheduler="none",
        cuda="0",
        lr=0.1,
        clip=2.0,
        batch_size=20,
        num_epochs=1000,
        hidden_dim=512,
        output_dim=256,
        num_gc_layers=2,
        dropout=0.0,
        method="base",
        name_suffix="",
        explainer_suffix="",
        align_steps=1000,
        explain_node=None,
        graph_idx=-1,
        mask_act="sigmoid",
        multigraph_class=-1,
        multinode_class=-1,
    )
    return parser.parse_args()

if __name__ == '__main__':
    prog_args = arg_parse()
    with open(os.path.join(prog_args.datadir,prog_args.graph_type,'label_map.txt'),'r',encoding='utf-8') as fin:
        label_map = eval(fin.read())
    label_map_txt = {0: '0 非投诉会话', 1: '1 客户感受差', 2: '2 投诉/曝光', 3: '3 推诿', 4: '4 未解决问题', 5: '5 四到类', 6: '6 人身伤害',
                 10: '10 重复提问', 99: '99 词', -1: '-1 未知标签'}
    labeltxt_map = {0: '未知标签', 1: '非投诉会话', 2: '客户感受差', 3: '投诉/曝光', 4: '词', 5: '未解决问题', 6: '四到类'}
    label_pred_cnt={}
    label_cnt = {}
    word_label = ''
    for k,v in labeltxt_map.items():
        if v == '词':
            word_label=k
        else:
            label_pred_cnt[v]=0
            label_cnt[v]=0
    if word_label == '':
        word_label= -999

    ckpt = io_utils.load_ckpt(prog_args)
    cg_dict = ckpt["cg"] # get computation graph
    pred = cg_dict["pred"]
    pred_idex = np.argmax(pred, axis=2)
    labels = cg_dict["label"]
    acc_sen = 0
    cnt_sen = 0
    fw_pred = open(os.path.join(prog_args.datadir,prog_args.graph_type,'pred_labels_all.txt'),'w',encoding='utf-8')
    with open(os.path.join(prog_args.datadir,prog_args.graph_type,'pred_labels_sen.txt'),'w',encoding='utf-8') as fw:
        for p,l in zip(pred_idex[0],labels[0]):
            fw_pred.write('\t'.join([str(label_map[p]),str(label_map[l])])+'\n')
            if l != word_label:
                label_pred_cnt[labeltxt_map[l]] += int(l==p)
                label_cnt[labeltxt_map[l]] += 1
                cnt_sen += 1
                acc_sen += int(l==p)
                fw.write('\t'.join([str(labeltxt_map[p]),str(labeltxt_map[l])])+'\n')
    fw_pred.close()
    print(f'句子数量为：{cnt_sen}, 正确数量为：{acc_sen}，准确率为{round(acc_sen/cnt_sen*100,2)}%')

    pred_idex_list,label_list = pred_idex[0],labels[0]
    precision, recall, f_score, true_sum = precision_recall_fscore_support(label_list, pred_idex_list,labels=[0, 1, 2, 3, 4, 5, 6])
    metrics_1 = classification_report(label_list, pred_idex_list, digits=3, labels=[0, 1, 2, 3, 4, 5, 6],
                                    target_names=[labeltxt_map[0], labeltxt_map[1], labeltxt_map[2],
                                                  labeltxt_map[3], labeltxt_map[4], labeltxt_map[5],
                                                  labeltxt_map[6]])
    print(metrics_1)
    metrics = {}
    for l,p,r,f in zip([0, 1, 2, 3, 4, 5, 6],precision,recall,f_score):
        # print(f'{label_map_txt[label_map[l]]}:  \n precesion:{p}  recall:{r}  f_score:{f}\n')
        metrics[labeltxt_map[l]] = {'precision':p,'recall':r,'f_score':f}

    df_metrics = pd.DataFrame.from_dict(metrics)
    print(df_metrics)
    # df_metrics.to_csv(os.path.join(prog_args.datadir, prog_args.graph_type, 'metrics.csv'))
    # with open(os.path.join(prog_args.datadir, prog_args.graph_type, 'metrics.txt'),'w',encoding='utf-8') as fw:
    #     print(f"metrics 保存路径{os.path.join(prog_args.datadir, prog_args.graph_type, 'metrics.txt')}")
    #     fw.write(str(metrics_1))
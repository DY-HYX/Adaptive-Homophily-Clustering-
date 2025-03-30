"""Main
"""
import argparse
import random
import scipy.sparse as sp
import torch
from models import HoLe
from models import HoLe_batch
from utils import get_str_time
from functions import get_data,normalize,set_seed,spixel_to_pixel_labels,cluster_accuracy,get_args,get_args_key,pprint_args,get_dataset
import LDA_SLIC_PU
import numpy as np
from sklearn.cluster import KMeans

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")#选择cpu或者GPU


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="HoLe",
        description=
        "Homophily-enhanced Structure Learning for Graph Clustering",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="Reddit",#Cora  BlogCatalog
        help="Dataset used in the experiment",
    )

    parser.add_argument('--dataset2', type=str, default='PaviaU',help='type of dataset.')  # 'Indian', 'Salinas', 'PaviaU'  'Houston','Trento'
    parser.add_argument('--superpixel_scale', type=int, default=130,help="superpixel_scale")  # IP 100 sa  250  pu160  Tr900  HU100
    parser.add_argument('--seed', type=int, default=5, help='Random seed.')#1  81.89  4 82.67  5 82.67

    args = parser.parse_args()


    dim = 300  #200 85.84
    n_lin_layers = 1
    dump = True
    set_seed(args.seed)
    lr = {
        "Cora": 0.001,
        "Citeseer": 0.001,
        "ACM": 0.001,
        "Pubmed": 0.001,
        "BlogCatalog": 0.001,
        "Flickr": 0.001,
        "Reddit": 1e-5,
    }
    n_gnn_layers = {
        "Cora": [8],
        "Citeseer": [3],
        "ACM": [3],
        "Pubmed": [35],
        "BlogCatalog": [1],
        "Flickr": [1],
        "Reddit": [3],#3
    }
    pre_epochs = {
        "Cora": [150],
        "Citeseer": [150],
        "ACM": [200],
        "Pubmed": [50],
        "BlogCatalog": [150],
        "Flickr": [300],
        "Reddit": [3],#3
    }
    epochs = {
        "Cora": 50,
        "Citeseer": 150,
        "ACM": 150,
        "Pubmed": 200,
        "BlogCatalog": 150,
        "Flickr": 150,
        "Squirrel": 150,
        "Reddit": 3,#3
    }
    inner_act = {
        "Cora": lambda x: x,
        "Citeseer": torch.sigmoid,
        "ACM": lambda x: x,
        "Pubmed": lambda x: x,
        "BlogCatalog": lambda x: x,
        "Flickr": lambda x: x,
        "Squirrel": lambda x: x,
        "Reddit": lambda x: x,
    }
    udp = {
        "Cora": 10,
        "Citeseer": 40,
        "ACM": 40,
        "Pubmed": 10,
        "BlogCatalog": 40,
        "Flickr": 40,
        "Squirrel": 40,
        "Reddit": 40,#40
    }
    node_ratios = {
        "Cora": [1],
        "Citeseer": [0.3],
        "ACM": [0.3],
        "Pubmed": [0.5],
        "BlogCatalog": [1],
        "Flickr": [0.3],
        "Squirrel": [0.3],
        "Reddit": [0.01],#0.01
    }
    add_edge_ratio = {
        "Cora": 0.5,
        "Citeseer": 0.5,
        "ACM": 0.5,
        "Pubmed": 0.5,
        "BlogCatalog": 0.5,
        "Flickr": 0.5,
        "Reddit": 0.005,
    }
    del_edge_ratios = {
        "Cora": [0.01],
        "Citeseer": [0.005],
        "ACM": [0.005],
        "Pubmed": [0.005],
        "BlogCatalog": [0.005],
        "Flickr": [0.005],
        "Reddit": [0.02],
    }
    gsl_epochs_list = {
        "Cora": [5],
        "Citeseer": [5],
        "ACM": [10],
        "Pubmed": [3],
        "BlogCatalog": [10],
        "Flickr": [10],
        "Reddit": [1],
    }
    regularization = {
        "Cora": 1,
        "Citeseer": 0,
        "ACM": 0,
        "Pubmed": 1,
        "BlogCatalog": 0,
        "Flickr": 0,
        "Reddit": 0,
    }
    source = {
        "Cora": "dgl",
        "Citeseer": "dgl",
        "ACM": "sdcn",
        "Pubmed": "dgl",
        "BlogCatalog": "cola",
        "Flickr": "cola",
        "Reddit": "cola",#dgl
    }

    datasets = [args.dataset]
    graph = {}

    for ds in datasets:

        hole = HoLe

        for gsl_epochs in gsl_epochs_list[ds]:
            runs = 10

            for n_gnn_layer in n_gnn_layers[ds]:
                for pre_epoch in pre_epochs[ds]:
                    for del_edge_ratio in del_edge_ratios[ds]:
                        for node_ratio in node_ratios[ds]:


                            time_name = get_str_time()


                            #############################################################################################
                            input, num_classes, y_true, gt_reshape, gt_hsi = get_data(args.dataset2)
                            # normalize data by band norm
                            input_normalize = normalize(input)
                            height, width, band = input_normalize.shape  # 145*145*200
                            print("height={0},width={1},band={2}".format(height, width, band))
                            input_numpy = np.array(input_normalize)

                            ls = LDA_SLIC_PU.LDA_SLIC(input_numpy, gt_hsi, num_classes - 1)
                            Q, S, A_SP, Edge_index, Edge_atter, Seg, A_ones = ls.simple_superpixel(
                                scale=args.superpixel_scale)
                            A_ones=sp.csr_matrix(A_ones)
                            Edge_index=tuple(torch.from_numpy(Edge_index))
                            S=torch.from_numpy(S)
                            #############################################################################################

                            graph["feat"]=S
                            graph["adj"] = A_ones
                            graph["edges"] = Edge_index
                            graph["edges"] = gt_reshape

                            features = S
                            # if ds in ("Cora", "Pubmed"):
                            #     graph.ndata["feat"][(features -
                            #                          0.0) > 0.0] = 1.0
                            adj_csr = A_ones
                            adj_sum_raw = adj_csr.sum()

                            edges = Edge_index
                            features_lil = sp.lil_matrix(features)



                            for run_id in range(runs):


                                model = hole(
                                    hidden_units=[dim],
                                    in_feats=features.shape[1],
                                    n_clusters=num_classes,
                                    n_gnn_layers=n_gnn_layer,
                                    n_lin_layers=n_lin_layers,
                                    lr=lr[ds],
                                    n_pretrain_epochs=pre_epoch,
                                    n_epochs=epochs[ds],
                                    norm="sym",
                                    renorm=True,
                                    tb_filename=
                                    f"{ds}_gnn_{n_gnn_layer}_node_{node_ratio}_{add_edge_ratio[ds]}_{del_edge_ratio}_gsl_{gsl_epochs}_pre_ep{pre_epoch}_ep{epochs[ds]}_dim{dim}_{random.randint(0, 999999)}",
                                    inner_act=inner_act[ds],
                                    udp=udp[ds],
                                    regularization=regularization[ds],

                                )

                                model.fit(
                                    graph=graph,
                                    device=device,
                                    add_edge_ratio=add_edge_ratio[ds],
                                    node_ratio=node_ratio,
                                    del_edge_ratio=del_edge_ratio,
                                    gsl_epochs=gsl_epochs,
                                    labels=gt_reshape,
                                    adj_sum_raw=adj_sum_raw,
                                    load=True,
                                    dump=dump,
                                )

                                with torch.no_grad():
                                    z_detached = model.get_embedding()
                                    Z = model.get_Q(z_detached)
                                    q = Z.detach().data.cpu().numpy().argmax(1)



                                predict_labels = KMeans(n_clusters=num_classes).fit_predict(z_detached.cpu().detach().numpy())
                                indx = np.where(gt_reshape != 0)
                                labels = gt_reshape[indx]
                                pixel_y = spixel_to_pixel_labels(predict_labels, Q)
                                prediction = pixel_y[indx]
                                acc, kappa, nmi, ari, pur, ca = cluster_accuracy(labels, prediction, return_aligned=False)
                                print( '\nz_detached: acc3: {:.2f}, nmi3: {:.2f}, ari3: {:.2f}'.format(acc * 100, nmi, ari))


                                predict_labels = KMeans(n_clusters=num_classes).fit_predict(Z.cpu().detach().numpy())
                                indx = np.where(gt_reshape != 0)
                                labels = gt_reshape[indx]
                                pixel_y = spixel_to_pixel_labels(predict_labels, Q)
                                prediction = pixel_y[indx]
                                acc, kappa, nmi, ari, pur, ca = cluster_accuracy(labels, prediction, return_aligned=False)
                                print( '\nQ : acc3: {:.2f}, nmi3: {:.2f}, ari3: {:.2f}'.format(acc * 100, nmi, ari))






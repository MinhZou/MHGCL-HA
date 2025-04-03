import numpy as np
import torch
import scipy.io as sio
import dgl
import pandas as pd
import torch.nn.functional as F

def one_hot_encode(x, n_classes):
    """
    One hot encode a list of sample labels. Return a one-hot encoded vector for each label.
    : x: List of sample Labels
    : return: Numpy array of one-hot encoded labels
     """
    return np.eye(n_classes)[x]


def load_network_data(name):
    net = sio.loadmat('./data/' + name + '.mat')
    X, A, Y = net['attrb'], net['network'], net['group']
    if name in ['cs', 'photo']:
        Y = Y.flatten()
        Y = one_hot_encode(Y, Y.max() + 1).astype(np.int32)
    return A, X, Y


def random_planetoid_splits(num_classes, y, train_num, seed):
    # Set new random planetoid splits:
    # *  train_num * num_classes labels for training
    # * 500 labels for validation
    # * 1000 labels for testing

    np.random.seed(seed)
    indices = []

    for i in range(num_classes):
        index = (y == i).nonzero().view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

    train_index = torch.cat([i[:train_num] for i in indices], dim=0)

    rest_index = torch.cat([i[train_num:] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0))]

    val_index = rest_index[:500]
    test_index = rest_index[500:1500]

    return train_index, val_index, test_index


def get_train_data(labels, tr_num, val_num, seed):
    np.random.seed(seed)
    labels_vec = labels.argmax(1)
    labels_num = labels_vec.max() + 1

    idx_train = []
    idx_val = []
    for label_idx in range(labels_num):
        pos0 = np.argwhere(labels_vec == label_idx).flatten()
        pos0 = np.random.permutation(pos0)
        idx_train.append(pos0[0:tr_num])
        idx_val.append(pos0[tr_num:val_num + tr_num])

    idx_train = np.array(idx_train).flatten()
    idx_val = np.array(idx_val).flatten()
    idx_test = np.setdiff1d(range(labels.shape[0]), np.union1d(idx_train, idx_val))

    idx_train = torch.LongTensor(np.random.permutation(idx_train))
    idx_val = torch.LongTensor(np.random.permutation(idx_val))
    idx_test = torch.LongTensor(np.random.permutation(idx_test))

    return idx_train, idx_val, idx_test


# def load_new_data(name):
#     pd_edge = pd.read_table("./data/" + name + "/out1_graph_edges.txt", sep='\t')
#     pd_feature = pd.read_table("./data/" + name + "/out1_node_feature_label.txt", sep='\t')
#
#     # create G by dgl
#     u = pd_edge['node_id']
#     v = pd_edge['node_id.1']
#
#     u = list(map(int, u))
#     v = list(map(int, v))
#     G = dgl.graph((u, v))
#
#     node_feature, node_label = pd_feature['feature'], pd_feature['label']
#
#     # class : 5, n_node : 2277
#     node_label = torch.tensor(node_label)
#     Y = F.one_hot(node_label)
#
#     # create tensor_feature :2277*2325,feature dim is 2325
#     feature_list = []
#     for i in range(node_feature.size):
#         tmp = list(map(int, node_feature[i].split(',')))
#         feature_list.append(tmp)
#     X = torch.tensor(feature_list,dtype=torch.float)
#
#     # create tensor_adj : 2277*2277
#     A = G.adjacency_matrix(transpose=True).to_dense()
#     # A = G.adjacency_matrix(transpose=True)
#     return A, X, Y, G



def drop_feature(x, drop_prob):
    drop_mask = torch.empty(
        (x.size(1), ),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0
    return x


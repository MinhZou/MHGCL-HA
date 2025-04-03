# -*- coding: utf-8 -*-
import argparse
import numpy as np
import time
import random
import torch
import torch.nn.functional as F
from unigencoder import UniGEncoder
from mhgcl import MHGCL

from utils import load_network_data, get_train_data, random_planetoid_splits
from loss import multihead_contrastive_loss
import warnings
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops, add_self_loops, to_dense_adj

import copy

import datetime

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='MHGCL')
parser.add_argument("--model", type=str, default="MHGCL",
                    help="model")
parser.add_argument("--cuda", type=int, default=0,
                    help="which GPU to use. Set -1 to use CPU.")
parser.add_argument("--epochs", type=int, default=2000,
                    help="number of training epochs")
parser.add_argument("--dataset", type=str, default="cora",  # citeseer, cs photo citeseer pubmed
                    help="which dataset for training")
parser.add_argument("--num_heads", type=int, default=4,
                    help="number of hidden attention heads")
parser.add_argument("--num_layers", type=int, default=1,
                    help="number of hidden layers")
parser.add_argument("--hidden", type=int, default=32,
                    help="number of hidden units")
parser.add_argument("--tau", type=float, default=1,
                    help="temperature-scales")
parser.add_argument("--seed", type=int, default=1,
                    help="random seed")
parser.add_argument("--dropout", type=float, default=0.6,
                    help="feature dropout")
parser.add_argument("--attn_drop", type=float, default=0.5,
                    help="attention dropout")
parser.add_argument("--lr", type=float, default=0.01,
                    help="learning rate")
parser.add_argument('--weight-decay', type=float, default=1e-4,
                    help="weight decay")
parser.add_argument('--negative-slope', type=float, default=0.2,
                    help="the negative slope of leaky relu")
parser.add_argument('--num_features', type=int, default=1,
                    help="aa")
parser.add_argument('--num_classes', type=int, default=1,
                    help="aadsa")
parser.add_argument('--norm_type', default=0, type=int)
parser.add_argument('--init_val', default=1, type=float)
parser.add_argument('--init_type', default=0, type=int)
parser.add_argument('--loss_type', default=2, type=int) 
parser.add_argument('--aug_type', default='hop_2v', type=str)
parser.add_argument('--adj_type', default=0, type=int)


args = parser.parse_args()

if args.dataset == 'cora':
    args.num_heads = 8
    args.hidden = 32
    args.tau = 0.5
    args.attn_drop = 0.9
    args.dropout = 0.4
    args.lr = 0.02
    args.weight_decay = 0.00005
elif args.dataset == 'citeseer':
    args.num_heads = 4
    args.hidden = 32
    args.tau = 10.0
    args.attn_drop = 0.8
    args.dropout = 0.4
    args.lr = 0.1
    args.weight_decay = 0.00005
elif args.dataset == 'cs':
    args.num_heads = 3
    args.hidden = 128
    args.tau = 0.1
    args.attn_drop = 0.6
    args.dropout = 0.9
    args.lr = 0.001
    args.weight_decay = 0.0
elif args.dataset == 'photo':
    args.num_heads = 2
    args.hidden = 128
    args.tau = 2.0
    args.attn_drop = 0.7
    args.dropout = 0.3
    args.lr = 0.02
    args.weight_decay = 0.00001
elif args.dataset == 'pubmed':
    args.num_heads = 3
    args.hidden = 32
    args.tau = 2.0
    args.attn_drop = 0.3
    args.dropout = 0.4
    args.lr = 0.001
    args.weight_decay = 0.0001
    
save_date = '12_01'
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

adj2, features, Y = load_network_data(args.dataset)

nonzero_rows, nonzero_cols = adj2.nonzero()

# Extract row and column indices
row_indices = torch.from_numpy(nonzero_rows).long()
col_indices = torch.from_numpy(nonzero_cols).long()
print('num edge:', len(row_indices), len(col_indices))

edge_index = torch.stack([row_indices, col_indices], dim=0)


features[features > 0] = 1

if args.cuda in [0, 1, 2, 3, 4, 5, 6, 7]:
    device = torch.device('cuda:' + str(args.cuda)
                          if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device('cpu')


features = torch.FloatTensor(features.todense())

f = open('{}_{}_{}_{}_{}.txt'.format(args.model, args.aug_type, args.adj_type, save_date, args.dataset), 'a+')
f.write('\n\n\n{}\n'.format(args))
f.flush()

labels = np.argmax(Y, 1)

all_time = time.time()
num_feats = features.shape[1]
n_classes = Y.shape[1]

features = torch.FloatTensor(features)
labels = torch.LongTensor(labels)
data = Data(x=features,
            edge_index=edge_index,
            y=labels)

N_vertex, edge_dict, node_num_egde = UniGEncoder.d_expansion(data)
pv_rows, pv_cols, pv_init_val, n_count = UniGEncoder.get_pval(edge_dict, node_num_egde, args.init_val, args.init_type)

edge_index_no_self_loops = remove_self_loops(edge_index)[0]
edge_index_with_self_loops = add_self_loops(edge_index)[0]
adj_matrix = to_dense_adj(edge_index_with_self_loops)[0]

if args.adj_type == 0:
    # # or the iteration version 
    data.adj = UniGEncoder.get_adj(N_vertex, pv_rows, pv_cols, pv_init_val, n_count, args.norm_type).to_dense()
    # # save memory and time
    # data.adj = []
    # adj = UniGEncoder.get_adj(N_vertex, pv_rows, pv_cols, pv_init_val, n_count, args.norm_type).to_dense()
    # for i in range(args.num_heads):
    #     tmp_adj = adj.clone()
    #     adj = torch.mm(tmp_adj, adj)
    #     data.adj.append(tmp_adj)
elif args.adj_type == 1:
    data.adj = adj_matrix
else:
    deg = adj_matrix.sum(dim=1)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    norm_adj_matrix = deg_inv_sqrt[:,None] * adj_matrix * deg_inv_sqrt
    data.adj = norm_adj_matrix

args.num_features = num_feats
args.num_classes = n_classes
args.num_nodes = features.shape[0]



model = MHGCL(args)

model = model.to(device)
data = data.to(device)

# use optimizer
optimizer = torch.optim.Adam(
    model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

# initialize graph
dur = []
test_acc = 0

counter = 0
min_train_loss = 100
early_stop_counter = 100
best_t = -1

for epoch in range(args.epochs):
    if epoch >= 0:
        t0 = time.time()
    model.train()
    optimizer.zero_grad()
    heads = model(data)

    loss = multihead_contrastive_loss(heads, tau=args.tau, loss_type=args.loss_type)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        heads = model(data)
        loss_train = multihead_contrastive_loss(heads, tau=args.tau, loss_type=args.loss_type)

    # early stop if loss does not decrease for 100 consecutive epochs
    if loss_train < min_train_loss:
        counter = 0
        min_train_loss = loss_train
        best_t = epoch
        torch.save(model.state_dict(), 'best_mhgcl.pkl')
    else:
        counter += 1

    if counter >= early_stop_counter:
        print('early stop')
        break

    if epoch >= 0:
        dur.append(time.time() - t0)

    print("Epoch {:04d} | Time(s) {:.4f} | TrainLoss {:.4f} ".
          format(epoch + 1, np.mean(dur), loss_train.item()))

print('Loading {}th epoch'.format(best_t))
model.load_state_dict(torch.load('best_mhgcl.pkl'))
model.eval()
with torch.no_grad():
    heads = model(data)

embeds = torch.cat(heads, axis=1)  # concatenate emb learned by all heads
embeds = embeds.detach().cpu()

Accuaracy_test_allK = []
numRandom = 20

for train_num in [1, 2, 3, 4, 20]:

    AccuaracyAll = []
    for random_state in range(numRandom):
        print(
            "\n=============================%d-th random split with training num %d============================="
            % (random_state + 1, train_num))

        if train_num == 20:
            if args.dataset in ['cora', 'citeseer', 'pubmed']:
                # train_num per class: 20, val_num: 500, test: 1000
                val_num = 500
                idx_train, idx_val, idx_test = random_planetoid_splits(n_classes, torch.tensor(labels), train_num,
                                                                       random_state)
            else:
                # Coauthor CS, Amazon Computers, Amazon Photo
                # train_num per class: 20, val_num per class: 30, test: rest
                val_num = 30
                idx_train, idx_val, idx_test = get_train_data(Y, train_num, val_num, random_state)

        else:
            val_num = 0  # do not use a validation set when the training labels are extremely limited
            idx_train, idx_val, idx_test = get_train_data(Y, train_num, val_num, random_state)

        train_embs = embeds[idx_train, :]
        val_embs = embeds[idx_val, :]
        test_embs = embeds[idx_test, :]

        if train_num == 20:
            # find the best parameter C using validation set
            best_val_score = 0.0
            for param in [1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100]:
                LR = LogisticRegression(solver='liblinear', multi_class='ovr', random_state=0, C=param)
                LR.fit(train_embs, labels[idx_train])
                val_score = LR.score(val_embs, labels[idx_val])
                if val_score > best_val_score:
                    best_val_score = val_score
                    best_parameters = {'C': param}

            LR_best = LogisticRegression(solver='liblinear', multi_class='ovr', random_state=0, **best_parameters)

            LR_best.fit(train_embs, labels[idx_train])
            y_pred_test = LR_best.predict(test_embs)  # pred label
            print("Best accuracy on validation set:{:.4f}".format(best_val_score))
            print("Best parameters:{}".format(best_parameters))

        else:  # not use a validation set when the training labels are extremely limited
            LR = LogisticRegression(solver='liblinear', multi_class='ovr', random_state=0)
            LR.fit(train_embs, labels[idx_train])
            y_pred_test = LR.predict(test_embs)  # pred label

        test_acc = accuracy_score(labels[idx_test], y_pred_test)
        print("test accuaray:{:.4f}".format(test_acc))
        AccuaracyAll.append(test_acc)

    average_acc = np.mean(AccuaracyAll) * 100
    std_acc = np.std(AccuaracyAll) * 100
    print('avg accuracy over %d random splits: %.1f +/- %.1f, for train_num: %d, val_num:%d\n' % (
        numRandom, average_acc, std_acc, train_num, val_num))
    f.write('avg accuracy over %d random splits: %.1f +/- %.1f, for train_num: %d, val_num:%d\n' % (
        numRandom, average_acc, std_acc, train_num, val_num))

    f.flush()
    Accuaracy_test_allK.append(average_acc)


f.write('Time: ')
f.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

f.close()

import torch
import utils
import numpy as np

import torch.nn as nn
import torch.nn.functional as F

import math

from torch_scatter import scatter
from torch_geometric.utils import softmax


class UniGEncoder(nn.Module):
    @staticmethod
    def d_expansion(data):
        V, E = data.edge_index
        print(data.edge_index)
        print(len(V), len(E))

        edge_dict = {}
        ccc = 0
        edge_set = set()
        for i in range(len(V)):
            if (V[i].item(), E[i].item()) not in edge_set and (E[i].item(), V[i].item()) not in edge_set:
                edge_dict[ccc] = []
                edge_dict[ccc].append(V[i])
                edge_dict[ccc].append(E[i])
                edge_set.add((V[i].item(), E[i].item()))
                ccc += 1
        unique_lists = list(set(tuple(lst) for lst in edge_dict.values()))
        edge_dict = {i: list(unique_lists[i]) for i in range(len(unique_lists))}
        print('unique:', len(edge_dict))
        N_vertex = data.num_nodes
        N_hy = len(edge_dict)
        print('number nodes and edges:', N_vertex, N_hy)
        # print(edge_dict)
        self_set = set()
        # print(N_vertex, len(edge_dict), N_hyperedge)
        for key, val in edge_dict.items():
            if len(val) == 1:
                self_set.add(val[0])
        # print(len(self_set))
        # add self-loop
        if len(self_set) < N_vertex:
            # print(len(self_set))
            count = 0
            for i in range(N_vertex):
                if i not in self_set:
                    edge_dict[N_hy + count] = []
                    edge_dict[N_hy + count].append(i)
                    count += 1
        print(len(self_set), len(edge_dict))
        # print(edge_dict)
        node_num_egde = {}
        for key, val in edge_dict.items():
            for v in val:
                if v not in node_num_egde:
                    node_num_egde[v] = 0
                else:
                    node_num_egde[v] += 1
        # print("feature shape:", data.x.shape)
        return N_vertex, edge_dict, node_num_egde

    @staticmethod
    def get_pval(edge_dict, node_num_egde, init_val, init_type):
        pv_rows = []
        pv_cols = []
        n_count = 0
        pv_init_val = []
        for i in range(len(edge_dict)):
            res_idx = np.array(edge_dict.get(i, []), dtype=np.int32)
            for q, p in enumerate(res_idx):
                if len(res_idx) == 1:
                    if res_idx[0] in node_num_egde:
                        if init_type == 1:
                            pv_init_val.append(node_num_egde[res_idx[0]] * init_val)
                        else:
                            pv_init_val.append(init_val)
                        # print()
                    else:
                        pv_init_val.append(1)
                else:
                    pv_init_val.append(1)
                pv_rows.append(n_count)
                pv_cols.append(p)
                if q == (len(res_idx) - 1):
                    n_count += 1
        return pv_rows, pv_cols, pv_init_val, n_count

    @staticmethod
    def get_adj(N_vertex, pv_rows, pv_cols, pv_init_val, n_count, norm_type):
        pv_rows = torch.tensor(pv_rows)
        pv_cols = torch.tensor(pv_cols)
        pv_indices = torch.stack([pv_rows, pv_cols], dim=0)
        pv_values = torch.tensor(pv_init_val, dtype=torch.float32)
        # pv_values = torch.ones_like(pv_rows, dtype=torch.float32)
        # Pv = torch.sparse_coo_tensor(pv_indices, pv_values, size=[len(pv_rows), N_vertex])
        Pv = torch.sparse_coo_tensor(pv_indices, pv_values, size=[n_count, N_vertex])
        PvT = torch.sparse_coo_tensor(torch.stack([pv_cols, pv_rows], dim=0), pv_values, size=[N_vertex, n_count])
        # print(Pv.shape, PvT.shape)
        # data.PvT = PvT
        if norm_type == 0:
            Pv_col_sum = torch.sparse.sum(Pv, dim=1)
            Pv_diag_indices = Pv_col_sum.indices()[0]
            Pv_diag_values = Pv_col_sum.values()
            Pv_diag_values = torch.reciprocal(Pv_diag_values)
            Pv_diag = torch.sparse_coo_tensor(torch.stack([Pv_diag_indices, Pv_diag_indices]), Pv_diag_values,
                                              torch.Size([Pv_col_sum.shape[0], Pv_col_sum.shape[0]]))
            Pv_col_norm = torch.sparse.mm(Pv_diag, Pv)

            PvT_row_sum = torch.sparse.sum(PvT, dim=1)
            PvT_diag_indices = PvT_row_sum.indices()[0]
            PvT_diag_values = PvT_row_sum.values()
            PvT_diag_values = torch.reciprocal(PvT_diag_values)
            PvT_diag = torch.sparse_coo_tensor(torch.stack([PvT_diag_indices, PvT_diag_indices]), PvT_diag_values,
                                               torch.Size([PvT_row_sum.shape[0], PvT_row_sum.shape[0]]))
            # print(PvT.shape, PvT_diag.shape)
            PvT_row_norm = torch.sparse.mm(PvT_diag, PvT)
        elif norm_type == 1:
            Pv_col_sum = torch.sparse.sum(Pv, dim=0)
            Pv_diag_indices = Pv_col_sum.indices()[0]
            Pv_diag_values = Pv_col_sum.values()
            Pv_diag_values = torch.reciprocal(Pv_diag_values)
            Pv_diag = torch.sparse_coo_tensor(torch.stack([Pv_diag_indices, Pv_diag_indices]), Pv_diag_values,
                                              torch.Size([Pv_col_sum.shape[0], Pv_col_sum.shape[0]]))
            # print(Pv.shape, Pv_diag.shape)
            Pv_col_norm = torch.sparse.mm(Pv, Pv_diag)

            PvT_row_sum = torch.sparse.sum(PvT, dim=1)
            PvT_diag_indices = PvT_row_sum.indices()[0]
            PvT_diag_values = PvT_row_sum.values()
            PvT_diag_values = torch.reciprocal(PvT_diag_values)
            PvT_diag = torch.sparse_coo_tensor(torch.stack([PvT_diag_indices, PvT_diag_indices]), PvT_diag_values,
                                               torch.Size([PvT_row_sum.shape[0], PvT_row_sum.shape[0]]))
            # print(PvT.shape, PvT_diag.shape)
            PvT_row_norm = torch.sparse.mm(PvT_diag, PvT)
        elif norm_type == 2:
            Pv_col_sum = torch.sparse.sum(Pv, dim=0)
            Pv_diag_indices = Pv_col_sum.indices()[0]
            Pv_diag_values = Pv_col_sum.values()
            Pv_diag_values = torch.reciprocal(Pv_diag_values)
            Pv_diag = torch.sparse_coo_tensor(torch.stack([Pv_diag_indices, Pv_diag_indices]), Pv_diag_values,
                                              torch.Size([Pv_col_sum.shape[0], Pv_col_sum.shape[0]]))
            # print(Pv.shape, Pv_diag.shape)
            Pv_col_norm = torch.sparse.mm(Pv, Pv_diag)

            PvT_row_sum = torch.sparse.sum(PvT, dim=0)
            PvT_diag_indices = PvT_row_sum.indices()[0]
            PvT_diag_values = PvT_row_sum.values()
            PvT_diag_values = torch.reciprocal(PvT_diag_values)
            PvT_diag = torch.sparse_coo_tensor(torch.stack([PvT_diag_indices, PvT_diag_indices]), PvT_diag_values,
                                               torch.Size([PvT_row_sum.shape[0], PvT_row_sum.shape[0]]))
            # print(PvT.shape, PvT_diag.shape)
            PvT_row_norm = torch.sparse.mm(PvT, PvT_diag)
        elif norm_type == 3:
            Pv_col_sum = torch.sparse.sum(Pv, dim=1)
            Pv_diag_indices = Pv_col_sum.indices()[0]
            Pv_diag_values = Pv_col_sum.values()
            Pv_diag_values = torch.reciprocal(Pv_diag_values)
            Pv_diag = torch.sparse_coo_tensor(torch.stack([Pv_diag_indices, Pv_diag_indices]), Pv_diag_values,
                                              torch.Size([Pv_col_sum.shape[0], Pv_col_sum.shape[0]]))
            # print(Pv.shape, Pv_diag.shape)
            Pv_col_norm = torch.sparse.mm(Pv_diag, Pv)

            PvT_row_sum = torch.sparse.sum(PvT, dim=0)
            PvT_diag_indices = PvT_row_sum.indices()[0]
            PvT_diag_values = PvT_row_sum.values()
            PvT_diag_values = torch.reciprocal(PvT_diag_values)
            PvT_diag = torch.sparse_coo_tensor(torch.stack([PvT_diag_indices, PvT_diag_indices]), PvT_diag_values,
                                               torch.Size([PvT_row_sum.shape[0], PvT_row_sum.shape[0]]))

            PvT_row_norm = torch.sparse.mm(PvT, PvT_diag)
        else:
            Pv_col_norm = Pv
            PvT_row_norm = PvT
        adj = torch.sparse.mm(PvT_row_norm, Pv_col_norm)
        from torch_geometric.utils.sparse import to_edge_index
        from torch_sparse import SparseTensor
        (row, col), val = to_edge_index(adj)
        # adj = SparseTensor(row=row, col=col, value=val, sparse_sizes=adj_shape)
        adj = SparseTensor(row=row, col=col, value=val, sparse_sizes=(N_vertex, N_vertex))
        return adj

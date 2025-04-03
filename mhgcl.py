import torch
import torch.nn as nn
import torch.nn.functional as F

#
class MHGCL(torch.nn.Module):
    def  __init__(self, args):
        super(MHGCL, self).__init__()
        self.num_head = args.num_heads
        self.hidden = args.hidden
        self.attn_drop = args.attn_drop
        self.dropout = args.dropout
        self.aug_type = args.aug_type
        self.adj_type = args.adj_type
        self.dataset = args.dataset
        if self.aug_type == 'hop_2v':
            self.lin = nn.Linear(args.num_features, args.hidden*2)
        else:
            self.lin = nn.Linear(args.num_features, args.hidden*self.num_head)
        # self.lins = [nn.Linear(args.num_features, args.hidden) for i in range(self.num_head)] separated linear

    def forward(self, data):
        x, edges = data.x, data.edge_index
        x = self.lin(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.attn_drop, training=self.training, inplace=True)
        if self.aug_type == 'hop_2v':
            x = x.view(-1, 2, self.hidden)
        else:
            x = x.view(-1, self.num_head, self.hidden)
        heads = []
        out = None
        adj = data.adj
        # from torch_sparse import matmul
        # x = matmul(adj, x)
        for i in range(self.num_head):
            if self.aug_type == 'hop':
                # the accuracy results are obtained by using the iteration version
                out = torch.mm(adj, x[:, i])
                adj = torch.mm(data.adj, adj)
                # save time 
                # out = torch.mm(adj[i], x[:, i])
                out = F.elu(out)
                out = F.dropout(out, p=self.dropout, training=self.training, inplace=True)
            elif self.aug_type == 'hop_2v':
                if i == 0:
                    out = torch.mm(adj, x[:, 0])
                    adj = torch.mm(data.adj, adj)
                    out = F.elu(out)
                    out = F.dropout(out, p=self.dropout, training=self.training, inplace=True)
                else:
                    if i == 1:
                        out = torch.mm(adj, x[:, 1])
                    else:
                        out = torch.mm(data.adj, out)
                    out = F.elu(out)
                    out = F.dropout(out, p=self.dropout, training=self.training, inplace=True)
            elif self.aug_type == 'none':
                out = x[:, i]
                out = F.elu(out)
                out = F.dropout(out, p=self.dropout, training=self.training, inplace=True)
            # saving memory MHGCL-HA(2V)
            if self.aug_type == 'hop_2v':
                if i == 0:
                    heads.append(out)
                if i == self.num_head - 1:
                    heads.append(out)
            # MHGCL-HA(MV)
            elif self.aug_type == 'hop':
                heads.append(out)
            else:
                raise ValueError(f"Unsupported aug_type: {self.aug_type}")
        return heads
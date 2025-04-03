# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
import numpy as np
import random


def nei_con_loss(z1: torch.Tensor, z2: torch.Tensor, tau, adj, hidden_norm: bool = True):
    '''neighbor contrastive loss'''
    adj = adj - torch.diag_embed(adj.diag())  # remove self-loop
    adj[adj > 0] = 1
    nei_count = torch.sum(adj, 1) * 2 + 1  # intra-view nei+inter-view nei+self inter-view
    nei_count = torch.squeeze(torch.tensor(nei_count))

    f = lambda x: torch.exp(x / tau)
    intra_view_sim = f(sim(z1, z1, hidden_norm))
    inter_view_sim = f(sim(z1, z2, hidden_norm))

    loss = (inter_view_sim.diag() + (intra_view_sim.mul(adj)).sum(1) + (inter_view_sim.mul(adj)).sum(1)) / (
            intra_view_sim.sum(1) + inter_view_sim.sum(1) - intra_view_sim.diag())
    # loss = intra_view_sim.diag() / (inter_view_sim.sum(1) + intra_view_sim.sum(1) - inter_view_sim.diag())

    loss = loss / nei_count  # divided by the number of positive pairs for each node

    return -torch.log(loss)


def contrastive_loss(z1: torch.Tensor, z2: torch.Tensor, adj,
                     mean: bool = True, tau: float = 1.0, hidden_norm: bool = True):
    l1 = nei_con_loss(z1, z2, tau, adj, hidden_norm)
    l2 = nei_con_loss(z2, z1, tau, adj, hidden_norm)
    ret = (l1 + l2) * 0.5
    ret = ret.mean() if mean else ret.sum()

    return ret


def multihead_contrastive_loss(heads, tau: float = 1.0, loss_type=0, loss_ratio=0.01, head_init=0):
    loss = torch.tensor(0, dtype=float, requires_grad=True)
    if loss_type == 0:
        for i in range(0, int(len(heads)//2)):
            loss = loss + contrastive_loss_node(heads[i], heads[i+int(len(heads)//2)], tau=tau)
        return loss / int(len(heads)//2)
    elif loss_type == 1:
        sim_loss = torch.nn.L1Loss()
        selected_loss = False
        selected_node = False
        if selected_loss:
            num_heads = [i for i in range(0, int(len(heads)//2))]
            num_to_select = 1
            selected_heads = random.sample(num_heads, num_to_select)

            for i in selected_heads:
                if i == 0:
                    loss = loss + contrastive_loss_node(heads[i], heads[i+int(len(heads)//2)], tau=tau)
                    loss = loss + contrastive_loss_node(heads[0], heads[1], tau=tau)
                else:
                    loss = loss + contrastive_loss_node(heads[i], heads[i+int(len(heads)//2)], tau=tau)
                    loss = loss + contrastive_loss_node(heads[0], heads[i], tau=tau)
            # return loss / (2*len(selected_heads))
            return loss / (2*num_to_select)
        elif selected_node:
            num_nodes = int(heads[0].size(0)*0.75)
            rand_idx = torch.randperm(heads[0].size(0))[:num_nodes]
            for i in range(0, int(len(heads)//2)):
                loss = loss + contrastive_loss_node(heads[i][rand_idx], heads[i+int(len(heads)//2)][rand_idx], tau=tau)
                if i >= 1:
                    loss = loss + contrastive_loss_node(heads[0][rand_idx], heads[i][rand_idx], tau=tau)
            return loss / (len(heads) - 1)
        else:
            for i in range(0, int(len(heads)//2)):
                loss = loss + contrastive_loss_node(heads[i], heads[i+int(len(heads)//2)], tau=tau)
                if i >= 1:
                    loss = loss + contrastive_loss_node(heads[0], heads[i], tau=tau)
            return loss / (len(heads) - 1)
    elif loss_type == 2:
        for i in range(1, len(heads)):
            loss = loss + contrastive_loss_node(heads[0], heads[i], tau=tau)
        return loss / (len(heads) - 1)
    elif loss_type == 20:
        head_nums = [i for i in range(len(heads))]
        head_nums.pop(head_init)
        for i in head_nums:
            loss = loss + contrastive_loss_node(heads[head_init], heads[i], tau=tau)
        return loss / (len(heads) - 1)
    elif loss_type == 30:
        head_nums = [i for i in range((len(heads)))]
        random.shuffle(head_nums)
        pairs = [[head_nums[i], head_nums[(i + 1) % len(head_nums)]] for i in range(len(head_nums))]
        for pair in pairs:
            loss = loss + contrastive_loss_node(heads[pair[0]], heads[pair[1]], tau=tau)
        return loss / (len(heads) - 1)
    elif loss_type == 3:
        for i in range(0, int(len(heads) // 2)):
            loss = loss + contrastive_loss_node(heads[i], heads[i+int(len(heads)//2)], tau=tau)
        return loss / (len(heads)//2)
    else:
        num_heads = [i for i in range(1, (len(heads)))]
        num_to_select = len(heads) // 2
        # selected_heads = random.sample(num_heads, num_to_select)
        selected_heads = num_heads
        for i in selected_heads:
            # if i >= 1:
            loss = loss + contrastive_loss_node(heads[0], heads[i], tau=tau)
        return loss / (len(selected_heads)-1)


def sim(z1: torch.Tensor, z2: torch.Tensor, hidden_norm: bool = True):
    if hidden_norm:
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
    return torch.mm(z1, z2.t())


def semi_loss(z1: torch.Tensor, z2: torch.Tensor, T):
    f = lambda x: torch.exp(x / T)
    refl_sim = f(sim(z1, z1))
    between_sim = f(sim(z1, z2))
    return -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))


def contrastive_loss_node(x1, x2, tau, com_nodes=None):
    T = tau
    l1 = semi_loss(x1, x2, T)
    l2 = semi_loss(x2, x1, T)
    ret = (l1 + l2) * 0.5
    ret = ret.mean()

    return ret






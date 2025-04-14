import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import random

     
class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, query, positive, negatives):
        B, qT, D = query.shape
        B1, T1, D = positive.shape

        #print('Query {}, Positive {}, Negatives {}'.format(query.shape, positive.shape, negatives.shape))
        B2, T2, E, D = negatives.shape  # B, N, D
        assert B == B1, f'Batch size mismatch: {B} != {B1}'
        assert B == B2, f'Batch size mismatch: {B} != {B2}'

        query = query.reshape(B*qT, D)
        positive = positive.reshape(B1*T1, D)
        negatives = negatives.reshape(B2*T2, -1, D)

        # Normalize embeddings
        query = F.normalize(query, dim=-1)      # (E, D)
        positive = F.normalize(positive, dim=-1) # (E, D)
        negatives = F.normalize(negatives, dim=-1) # (E, N, D)

        # Compute similarity scores
        pos_sim = torch.exp(torch.sum(query * positive, dim=-1) / self.temperature)  # (B,)
        neg_sim = torch.exp(torch.sum(query.unsqueeze(1) * negatives, dim=-1) / self.temperature)  # (B, N)

        # Compute InfoNCE loss
        #print(f'pos_sim {pos_sim[0:5]} \nneg_sim {neg_sim.sum(dim=-1)[0:5]}')
        denominator = pos_sim + neg_sim.sum(dim=-1)  # (B,)
        loss = -torch.log(pos_sim / denominator).mean()
        return loss
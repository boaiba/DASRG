import torch
import torch.nn as nn
import torch.nn.functional as F
from .models import (
    FeaExtra,
    AsyncGraph,
    SyncGraph,
    Aggregate,
)


class ASG(nn.Module):

    def __init__(self, config):
        super(ASG, self).__init__()
        self.config = config
        self.fea_extra = FeaExtra(config)
        self.ar_graph = AsyncGraph(config)
        self.sr_graph = SyncGraph(config)
        self.graph_agg = Aggregate(config)
        self.classifier = nn.Sequential(
            nn.Linear(config.fea_dim * 4 * config.num_ind, config.fea_dim * 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.fea_dim * 2, config.fea_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.fea_dim, config.num_group_act)
        )
        self.init_wts()

    def init_wts(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.constant_(param.data, 0)

    def forward(self, x, coord, person_y=None):

        # fea extraction
        ind_fea = self.fea_extractor(x)

        # AsyncGraph
        async_fea = self.ar_graph(ind_fea, coord, person_y)

        # SyncGraph
        avg_fea = torch.mean(ind_fea, dim=2)
        sync_fea = self.sr_graph(avg_fea, coord)

        async_adj = self.async_adj(async_fea, coord)
        sync_adj = self.sync_adj(coord)

        # Aggregate
        agg_fea = self.graph_agg(
            async_fea, sync_fea, async_adj, sync_adj
        )

        global_fea = agg_fea.view(x.size(0), -1)

        logit = self.classifier(global_fea)

        return logit

    def async_adj(self, async_fea, coord):
        batch_size, ind, _ = async_fea.shape
        device = async_fea.device

        adj = torch.zeros(batch_size, ind, ind, device=device)

        for b in range(batch_size):
            for i in range(ind):
                for j in range(ind):
                    if i != j:
                        fea_sim = F.cosine_similarity(
                            async_fea[b, i].unsqueeze(0),
                            async_fea[b, j].unsqueeze(0)
                        )

                        dist = torch.norm(coord[b, i] - coord[b, j])
                        dist_weight = torch.exp(-dist / self.config.dist_thd)

                        adj[b, i, j] = fea_sim * dist_weight

        return adj

    def sync_adj(self, coord):
        """ adj matrix"""
        batch_size, ind, _ = coord.shape
        device = coord.device

        adj = torch.zeros(batch_size, ind, ind, device=device)

        for b in range(batch_size):
            for i in range(ind):
                for j in range(ind):
                    if i != j:
                        dist = torch.norm(coord[b, i] - coord[b, j])
                        if dist < self.config.dist_thd:
                            adj[b, i, j] = 1.0

        return adj

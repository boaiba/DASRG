import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FeaExtra(nn.Module):
    def __init__(self, config):
        super(FeaExtra, self).__init__()
        self.config = config
        self.out_dim = config.feat_dim
        self.ch_ratio = config.channel_ratio

        # CNN for pocket and wrist
        self.p_net = self.cnn()
        self.w_net = self.cnn()

        # BLSTM for pocket and wrist
        self.p_lstm = nn.LSTM(input_size=config.cnn_out_dim, hidden_size=config.lstm_hid_dim,
                                   num_layers=1, batch_first=True, bidirectional=True)
        self.w_lstm = nn.LSTM(input_size=config.cnn_out_dim, hidden_size=config.lstm_hid_dim,
                                  num_layers=1, batch_first=True, bidirectional=True)

        self.merge_layer = nn.Sequential(
            nn.Linear(config.lstm_hid_dim * 4, self.out_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )

    def cnn(self):
        return nn.Sequential(
            nn.Conv1d(6, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),

            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        b, n, t, c, w = x.shape
        x = x.view(-1, c, w)

        p_out = self.p_net(x[:, :6, :])
        w_out = self.w_net(x[:, 6:, :])

        p_out = p_out.mean(dim=2)
        w_out = w_out.mean(dim=2)

        p_out = p_out.view(b * n, t, -1)
        w_out = w_out.view(b * n, t, -1)

        p_seq, _ = self.p_lstm(p_out)
        w_seq, _ = self.w_lstm(w_out)

        p_fea = p_seq.mean(dim=1)
        w_fea = w_seq.mean(dim=1)

        # feature fusion
        combined = torch.cat([p_fea, w_fea], dim=1)
        final_fea = self.merge_layer(combined)
        final_fea = final_fea.view(b, n, t, -1)

        return final_fea


class AsyncGraph(nn.Module):
    def __init__(self, config):
        super(AsyncGraph, self).__init__()
        self.config = config

        self.node_classifier = nn.Linear(config.fea_dim, config.num_ind_actions)
        self.proj_a = nn.Linear(config.fea_dim, config.fea_dim)
        self.proj_b = nn.Linear(config.fea_dim, config.fea_dim)
        self.proj_c = nn.Linear(config.fea_dim, config.fea_dim)
        self.edge_classifier = nn.Linear(config.fea_dim, config.num_ind_actions)

        self.map_real = nn.Linear(config.fea_dim, config.fea_dim // config.channel_ratio)
        self.map_delay = nn.Linear(config.fea_dim, config.fea_dim - config.fea_dim // config.channel_ratio)
        self.comb_wt = nn.Parameter(torch.randn(config.fea_dim, config.fea_dim))

        # relation infer
        self.query_proj = nn.Linear(config.fea_dim, config.fea_dim)
        self.key_proj = nn.Linear(config.fea_dim, config.fea_dim)

        # graph convolution
        self.gconv_wt = nn.Parameter(torch.randn(config.fea_dim, config.fea_dim))

    def forward(self, fea_data, pos_data, labels=None):
        # delay detection
        lag_info, conn_mask = self.time_lags(fea_data, labels)
        # feature fusion with time alignment
        aligned_fea = self.align_fuse(fea_data, lag_info, conn_mask)
        adj_mat = self.build_relation(aligned_fea, pos_data, conn_mask)
        async_out = self.gconv(aligned_fea, adj_mat)
        return async_out

    def time_lags(self, fea_data, labels=None):
        batch_sz, num_people, num_wins, _ = fea_data.shape
        dev = fea_data.device

        lag_offsets = torch.zeros(batch_sz, num_people, num_people, dtype=torch.long, device=dev)
        conn_flags = torch.zeros(batch_sz, num_people, num_people, device=dev)

        for b in range(batch_sz):
            for p1 in range(num_people):
                for p2 in range(num_people):
                    if p1 != p2:
                        max_score = 0
                        best_lag = 0
                        # label consistency check
                        if labels is not None:
                            label_match = (labels[b, p1] == labels[b, p2]).float()
                        else:
                            label_match = 1.0
                        # try different time lags
                        for lag in self.config.time_delay:
                            score = self.calc_lag(
                                fea_data[b, p1], fea_data[b, p2], lag
                            )
                            boosted_score = score * (0.5 + 0.5 * label_match)
                            if boosted_score > max_score:
                                max_score = boosted_score
                                best_lag = lag

                        lag_offsets[b, p1, p2] = best_lag
                        conn_flags[b, p1, p2] = 1.0 if max_score > self.config.prob_thd else 0.0

        return lag_offsets, conn_flags

    def calc_lag(self, p1_fea, p2_fea, lag):
        win_cnt = p1_fea.shape[0]
        node_cost = torch.mean(self.node_classifier(p1_fea))
        total_edge_cost = 0
        valid_pairs = 0

        for w in range(win_cnt):
            w_shifted = w + lag
            if w_shifted < win_cnt:
                b_p1 = self.proj_b(p1_fea[w])
                c_p2 = self.proj_c(p2_fea[w_shifted])
                space_time_corr = torch.exp(torch.dot(b_p1, c_p2))
                a_p1 = self.proj_a(p1_fea[w])
                edge_pred = self.edge_classifier(a_p1)
                compat_score = 1.0
                edge_cost = space_time_corr * torch.mean(edge_pred) * compat_score
                total_edge_cost += edge_cost
                valid_pairs += 1

        if valid_pairs > 0:
            total_edge_cost /= valid_pairs

        energy_sum = node_cost + total_edge_cost
        prob_score = torch.sigmoid(-energy_sum)
        return prob_score.item()

    def align_fuse(self, fea_data, lag_offsets, conn_flags):
        batch_sz, num_people, num_wins, fea_dim = fea_data.shape
        dev = fea_data.device
        aligned_data = torch.zeros(batch_sz, num_people, fea_dim, device=dev)

        for b in range(batch_sz):
            for p1 in range(num_people):
                win_fea = []
                for w in range(num_wins):
                    cur_fea = fea_data[b, p1, w]
                    lag_fea = []
                    for p2 in range(num_people):
                        if conn_flags[b, p1, p2] > 0:
                            offset = lag_offsets[b, p1, p2].item()
                            w_lag = w + offset
                            p2_shifted = fea_data[b, p2, w_lag] if w_lag < num_wins else fea_data[b, p2, -1]
                            real_part = self.map_real(cur_fea)
                            delay_part = self.map_delay(p2_shifted)
                            fused_pair = torch.cat([real_part, delay_part], dim=0)
                            lag_fea.append(fused_pair)

                    if lag_fea:
                        stacked_lags = torch.stack(lag_fea)
                        mix_wts = F.softmax(torch.randn(len(lag_fea), device=dev), dim=0)
                        win_fused = torch.sum(stacked_lags * mix_wts.unsqueeze(1), dim=0)
                    else:
                        win_fused = cur_fea
                    win_fea.append(win_fused)

                aligned_data[b, p1] = torch.stack(win_fea).mean(dim=0)

        return aligned_data

    def build_relation(self, fea_data, pos_data, conn_flags):
        batch_sz, num_people, fea_dim = fea_data.shape
        dev = fea_data.device
        rel_mats = torch.zeros(batch_sz, num_people, num_people, device=dev)

        for b in range(batch_sz):
            for p1 in range(num_people):
                for p2 in range(num_people):
                    if p1 != p2:
                        q_proj = self.query_proj(fea_data[b, p1])
                        k_proj = self.key_proj(fea_data[b, p2])
                        action_sim = torch.dot(q_proj, k_proj) / math.sqrt(fea_dim)
                        pos_dist = torch.norm(pos_data[b, p1] - pos_data[b, p2])
                        spatial_wt = torch.exp(-pos_dist / self.config.dist_thd)
                        time_flag = conn_flags[b, p1, p2]
                        rel_mats[b, p1, p2] = spatial_wt * torch.exp(action_sim) * time_flag

            for p1 in range(num_people):
                if rel_mats[b, p1].sum() > 0:
                    rel_mats[b, p1] = F.softmax(rel_mats[b, p1], dim=0)

        return rel_mats

    def gconv(self, fea_data, rel_mats):
        # batch_sz, num_people, fea_dim = fea_data.shape
        deg_mats = torch.sum(rel_mats, dim=2, keepdim=True) + 1e-6
        deg_inv_sqrt = torch.pow(deg_mats, -0.5)
        norm_adj = deg_inv_sqrt * rel_mats * deg_inv_sqrt.transpose(1, 2)
        gconv_out = torch.matmul(norm_adj, fea_data)
        gconv_out = torch.matmul(gconv_out, self.gconv_wt)
        return F.relu(gconv_out)


class SyncGraph(nn.Module):
    def __init__(self, config):
        super(SyncGraph, self).__init__()
        self.config = config
        # interact
        self.pair_encoder = nn.Sequential(
            nn.Linear(config.fea_dim * 2, config.fea_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.fea_dim, config.fea_dim)
        )

        self.spatial_mixer = nn.Sequential(
            nn.Linear(config.fea_dim * 2, config.fea_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.fea_dim, config.fea_dim)
        )
        # residual relationship
        self.diff_wts = nn.Parameter(torch.randn(config.num_ind, config.fea_dim))
        self.diff_bias = nn.Parameter(torch.randn(config.num_ind))
        # graph convolution
        self.sync_gconv_wt = nn.Parameter(torch.randn(config.fea_dim * 2, config.fea_dim))

    def forward(self, fea_data, pos_data):
        spatial_fea = self.spatial_interact(fea_data)
        diff_fea = self.residual_diffs(fea_data)
        merged_fea = torch.cat([spatial_fea, diff_fea], dim=2)
        neighbor_mat = self.neighbor_mat(pos_data)
        sync_out = self.sync_gconv(merged_fea, neighbor_mat)
        return sync_out

    def spatial_interact(self, fea_data):
        batch_sz, num_people, fea_dim = fea_data.shape
        dev = fea_data.device
        spatial_out = torch.zeros_like(fea_data, device=dev)

        for b in range(batch_sz):
            for p1 in range(num_people):
                cur_fea = fea_data[b, p1]
                # interactions with other people
                pair_feats = [self.pair_encoder(torch.cat([cur_fea, fea_data[b, p2]])) for p2 in range(num_people)]
                interact_sum = torch.stack(pair_feats).sum(dim=0)
                merged_input = torch.cat([cur_fea, interact_sum])
                spatial_out[b, p1] = self.spatial_mixer(merged_input)

        return spatial_out

    def residual_diffs(self, fea_data):
        batch_sz, num_people, fea_dim = fea_data.shape
        dev = fea_data.device
        diff_out = torch.zeros_like(fea_data, device=dev)

        for b in range(batch_sz):
            for p1 in range(num_people):
                cur_fea = fea_data[b, p1]
                # soft assignment weights
                assign_wts = F.softmax(torch.stack([
                    torch.dot(self.diff_wts[p2], cur_fea) + self.diff_bias[p2] for p2 in range(num_people)
                ]), dim=0)
                # weighted residual features
                weighted_diff = sum(assign_wts[p2] * (cur_fea - fea_data[b, p2]) for p2 in range(num_people))
                diff_out[b, p1] = weighted_diff

        return diff_out

    def neighbor_mat(self, pos_data):
        batch_sz, num_people, _ = pos_data.shape
        dev = pos_data.device
        neighbor_mat = torch.zeros(batch_sz, num_people, num_people, device=dev)

        for b in range(batch_sz):
            for p1 in range(num_people):
                for p2 in range(num_people):
                    if p1 != p2 and torch.norm(pos_data[b, p1] - pos_data[b, p2]) < self.config.dist_thd:
                        neighbor_mat[b, p1, p2] = 1.0

        return neighbor_mat

    def sync_gconv(self, fea_data, neighbor_mat):
        # graph convolution operation
        batch_sz, num_people, fea_dim = fea_data.shape

        eye_mat = torch.eye(num_people, device=fea_data.device).unsqueeze(0).expand(batch_sz, -1, -1)
        neighbor_mat = neighbor_mat + eye_mat

        deg_mats = torch.sum(neighbor_mat, dim=2, keepdim=True) + 1e-6
        deg_inv_sqrt = torch.pow(deg_mats, -0.5)
        # symmetric normalization
        norm_adj = deg_inv_sqrt * neighbor_mat * deg_inv_sqrt.transpose(1, 2)

        gconv_out = torch.matmul(norm_adj, fea_data)
        gconv_out = torch.matmul(gconv_out, self.sync_gconv_wt)
        return F.relu(gconv_out)


class Aggregate(nn.Module):
    def __init__(self, config):
        super(Aggregate, self).__init__()
        self.config = config

        # multiHead attention for async and sync graphs
        self.async_attn = nn.MultiheadAttention(
            config.fea_dim, config.num_heads, dropout=config.dropout, batch_first=True
        )
        self.sync_attn = nn.MultiheadAttention(
            config.fea_dim, config.num_heads, dropout=config.dropout, batch_first=True
        )
        self.async_mapper = nn.Linear(config.fea_dim, config.fea_dim)
        self.sync_mapper = nn.Linear(config.fea_dim, config.fea_dim)

    def forward(self, async_fea, sync_fea, async_adj, sync_adj):

        # update async nodes using sync features
        refined_async = self.async_nodes(async_fea, sync_fea, sync_adj)

        # update sync nodes using async features
        refined_sync = self.sync_nodes(sync_fea, async_fea, async_adj)

        final_fea = torch.cat([
            refined_async, refined_sync, async_fea, sync_fea
        ], dim=2)

        return final_fea

    def async_nodes(self, async_fea, sync_fea, sync_adj):

        batch_sz, num_people, fea_dim = async_fea.shape
        dev = async_fea.device

        refined_fea = torch.zeros_like(async_fea, device=dev)

        for b in range(batch_sz):
            for async_idx in range(num_people):
                # sync nodes connected to this async node
                edge_sync = []
                edge_cnt = 0

                for sync_idx in range(num_people):
                    if sync_adj[b, sync_idx, async_idx] > 0:
                        edge_sync.append(sync_fea[b, sync_idx])
                        edge_cnt += 1

                if edge_cnt > 0:
                    # process connected sync node features
                    sync_stack = torch.stack(edge_sync)
                    sync_stack = sync_stack.unsqueeze(0)
                    attn_out, _ = self.async_attn(
                        sync_stack, sync_stack, sync_stack
                    )
                    attn_out = attn_out.squeeze(0)

                    pooled = torch.mean(attn_out, dim=0) / edge_cnt
                    refined_fea[b, async_idx] = self.async_mapper(pooled)
                else:
                    refined_fea[b, async_idx] = async_fea[b, async_idx]

        return refined_fea

    def sync_nodes(self, sync_fea, async_fea, async_adj):
        batch_sz, num_people, fea_dim = sync_fea.shape
        dev = sync_fea.device

        refined_fea = torch.zeros_like(sync_fea, device=dev)

        for b in range(batch_sz):
            for sync_idx in range(num_people):
                # async nodes connected to this sync node
                edge_async = []
                edge_cnt = 0

                for async_idx in range(num_people):
                    if async_adj[b, async_idx, sync_idx] > 0:
                        edge_async.append(async_fea[b, async_idx])
                        edge_cnt += 1

                if edge_cnt > 0:
                    # process connected async node features
                    async_stack = torch.stack(edge_async)
                    async_stack = async_stack.unsqueeze(0)

                    attn_out, _ = self.sync_attn(
                        async_stack, async_stack, async_stack
                    )
                    attn_out = attn_out.squeeze(0)
                    pooled = torch.mean(attn_out, dim=0) / edge_cnt
                    refined_fea[b, sync_idx] = self.sync_mapper(pooled)
                else:
                    refined_fea[b, sync_idx] = sync_fea[b, sync_idx]

        return refined_fea
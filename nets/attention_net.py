import math

import torch
import torch.nn as nn
import torch.nn.functional as F
# from nets.graph_encoder import GraphAttentionEncoder


class UserEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_heads=8, n_layers=6,
                 normalization='batch', feed_forward_hidden=512, dropout=0.5):
        super(UserEncoder, self).__init__()
        # self.transformer = GraphAttentionEncoder(n_heads, hidden_dim, n_layers,
        #                                          input_dim, normalization, feed_forward_hidden, dropout=dropout)
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        # h,  # (batch_size, graph_size, embed_dim)
        # h.mean(dim=1)  # average to get embedding of graph, (batch_size, embed_dim)
        # return self.transformer(inputs)
        return self.dropout(self.embedding(inputs))


class ServerEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_heads=8, n_layers=6,
                 normalization='batch', feed_forward_hidden=512, dropout=0.5):
        super(ServerEncoder, self).__init__()
        # self.transformer = GraphAttentionEncoder(n_heads, hidden_dim, n_layers,
        #                                          input_dim, normalization, feed_forward_hidden, dropout=dropout)
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        # h,  # (batch_size, graph_size, embed_dim)
        # h.mean(dim=1)  # average to get embedding of graph, (batch_size, embed_dim)
        # return self.transformer(inputs)
        return self.dropout(self.embedding(inputs))


class Glimpse(nn.Module):
    # input :
    # query:    batch_size * 1 * query_input_dim
    # ref:      batch_size * seq_len * ref_hidden_dim
    def __init__(self, input_dim, hidden_dim, dropout=0.5):
        super(Glimpse, self).__init__()
        self.q = nn.Linear(input_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)
        self._norm_fact = 1 / math.sqrt(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, ref):
        Q = self.q(query)  # Q: batch_size * 1 * hidden_dim
        K = self.k(ref)  # K: batch_size * seq_len * hidden_dim
        V = self.v(ref)  # V: batch_size * seq_len * hidden_dim

        attn = nn.Softmax(dim=-1)(
            torch.bmm(Q, K.permute(0, 2, 1))) * self._norm_fact  # Q * K.T() # batch_size * 1 * seq_len

        output = self.dropout(torch.bmm(attn, V))  # Q * K.T() * V # batch_size * 1 * hidden_dim
        # 混合了所有服务器的相似度的一个表示服务器的变量
        return output


class Attention(nn.Module):
    def __init__(self, hidden_dim, dropout=0.5):
        super(Attention, self).__init__()
        self.hidden_size = hidden_dim
        self.W1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.vt = nn.Linear(hidden_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, decoder_state, encoder_outputs, mask):
        # (batch_size, max_seq_len, hidden_size)
        encoder_transform = self.W1(encoder_outputs)

        # (batch_size, 1, hidden_size)
        decoder_transform = self.W2(decoder_state)

        # (batch_size, max_seq_len, 1) => (batch_size, max_seq_len)
        u_i = self.dropout(self.vt(torch.tanh(encoder_transform + decoder_transform)).squeeze(-1))

        # softmax with only valid inputs, excluding zero padded parts
        # log_softmax for a better numerical stability
        score = u_i.masked_fill(~mask, value=torch.log(torch.tensor(1e-45)))
        return score


class PointerNet(nn.Module):
    def __init__(self, user_input_dim, server_input_dim, hidden_dim, device, dropout=0.1, server_reward_rate=0.1,
                 policy='sample'):
        super(PointerNet, self).__init__()
        # decoder hidden size
        self.hidden_dim = hidden_dim
        self.device = device

        self.user_encoder = UserEncoder(user_input_dim, hidden_dim, dropout=dropout).to(device)
        self.server_encoder = ServerEncoder(server_input_dim + 1, hidden_dim, dropout=dropout).to(device)

        # glimpse输入（用户，上次选择的服务器），维度为2*dim， 跟所有的服务器作相似度并输出融合后的服务器
        self.glimpse = Glimpse(hidden_dim, hidden_dim, dropout=dropout).to(device)
        self.pointer = Attention(hidden_dim, dropout=dropout).to(device)
        self.sm = nn.Softmax(dim=1).to(device)
        self.server_reward_rate = server_reward_rate
        self.policy = policy

    def forward(self, user_input_seq, server_input_seq, masks):
        batch_size = user_input_seq.size(0)
        user_len = user_input_seq.size(1)
        server_len = server_input_seq.size(1)

        # 真实分配情况
        user_allocate_list = -torch.ones(batch_size, user_len, dtype=torch.long, device=self.device)
        # 服务器分配矩阵，加一是为了给index为-1的来赋值
        server_allocate_mat = torch.zeros(batch_size, server_len + 1, dtype=torch.long, device=self.device)

        # 给服务器添加一个是否active位
        server_seq = torch.cat((server_input_seq, server_allocate_mat[:, :-1].unsqueeze(-1)), dim=-1)

        # encoder_output => (batch_size, max_seq_len, hidden_size) if batch_first
        # hidden_size is usually set same as embedding size
        # encoder_hidden => (num_layers * num_directions, batch_size, hidden_size) for each of h_n and c_n
        user_encoder_outputs = self.user_encoder(user_input_seq)
        server_encoder_outputs = self.server_encoder(server_seq)

        # last_chosen_server = torch.zeros(batch_size, 1, self.hidden_dim, device=self.device)
        action_probs = []
        action_idx = []

        # (batch_size, server_len 20, server_dim 4)
        tmp_server_capacity = server_seq[:, :, 3:7].clone()

        for i in range(user_len):
            mask = masks[:, i]

            user = user_encoder_outputs[:, i, :].unsqueeze(1)

            server_glimpse = self.glimpse(user, server_encoder_outputs)

            # get a pointer distribution over the encoder outputs using attention
            # (batch_size, server_len)
            logits = self.pointer(server_glimpse, server_encoder_outputs, mask)
            # (batch_size, server_len)
            probs = F.softmax(logits, dim=1)

            if self.policy == 'sample':
                # (batch_size, 1)
                idx = probs.multinomial(num_samples=1)
                prob = torch.gather(probs, dim=1, index=idx)
            elif self.policy == 'greedy':
                prob, idx = probs.topk(k=1)
            else:
                raise NotImplementedError

            prob = prob.squeeze(1)
            idx = idx.squeeze(1)
            action_probs.append(prob)
            action_idx.append(idx)

            # 第j个用户的分配服务器，取值只有0-19
            server_id = idx
            # 取出一个batch里所有第j个用户选择的服务器
            index_tensor = server_id.unsqueeze(-1).unsqueeze(-1).expand(batch_size, 1, 4)  # 4个资源维度
            j_th_server_capacity = torch.gather(tmp_server_capacity, dim=1, index=index_tensor).squeeze(1)
            # (batch_size)的True，False矩阵
            can_be_allocated = can_allocate(user_input_seq[:, i, 2:], j_th_server_capacity)
            # 如果不能分配容量就不减
            mask = can_be_allocated.unsqueeze(-1).expand(batch_size, 4)
            # 切片时指定batch和server_id对应关系
            batch_range = torch.arange(batch_size)
            # 服务器减去相应容量
            tmp_server_capacity[batch_range, server_id] -= user_input_seq[:, i, 2:] * mask
            # 记录服务器分配情况，即server_id和mask的内积
            server_id = torch.masked_fill(server_id, mask=~can_be_allocated, value=-1)
            # 真实分配情况
            user_allocate_list[:, i] = server_id
            # 给分配了的服务器在服务器分配矩阵中赋值为True
            server_allocate_mat[batch_range, server_id] = 1

            server_now = torch.cat((server_seq[:, :, :3], tmp_server_capacity,
                                    server_allocate_mat[:, :-1].unsqueeze(-1)),
                                   dim=-1)
            server_encoder_outputs = self.server_encoder(server_now)

        action_probs = torch.stack(action_probs)
        action_idx = torch.stack(action_idx, dim=-1)

        # 目前user_allocate_list是(batch_size, user_len)
        # 计算每个分配的用户数，即不是-1的个数，(batch_size)
        user_allocate_num = torch.sum(user_allocate_list != -1, dim=1)
        user_allocated_props = user_allocate_num.float() / user_len

        server_used_num = torch.sum(server_allocate_mat[:, :-1], dim=1)
        server_used_props = server_used_num.float() / server_len

        # 已使用的服务器的资源利用率
        original_servers_capacity = server_seq[:, :, 3:7]
        server_allocated_flag = server_allocate_mat[:, :-1].unsqueeze(-1).expand(batch_size, server_len, 4)
        used_original_server = original_servers_capacity.masked_fill(~server_allocated_flag.bool(), value=0)
        servers_remain_capacity = tmp_server_capacity.masked_fill(~server_allocated_flag.bool(), value=0)
        sum_all_capacity = torch.sum(used_original_server, dim=(1, 2))
        sum_remain_capacity = torch.sum(servers_remain_capacity, dim=(1, 2))
        capacity_used_props = 1 - sum_remain_capacity / sum_all_capacity

        # 原本的reward需要减这个 - server_used_props * self.server_reward_rate
        return -(user_allocated_props + capacity_used_props), \
            action_probs, action_idx, user_allocated_props, server_used_props, capacity_used_props, user_allocate_list


def can_allocate(workload: torch.Tensor, capacity: torch.Tensor):
    """
    计算能不能分配并返回分配情况
    :param workload: (batch, 4)
    :param capacity: (batch, 4)
    :return:
    """
    # (batch, 4)
    bools = capacity >= workload
    # (batch)，bool值
    return bools.all(dim=1)

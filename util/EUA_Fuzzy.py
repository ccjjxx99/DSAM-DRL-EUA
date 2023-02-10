import copy
import numpy as np

from util.utils import mask_trans_to_list

EL, VL, L, M, H, VH, EH = 0, 1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6, 1
omega_dic = {'ML': {"SL": EL, "SM": VL, "SH": VL},
             'MM': {"SL": M, "SM": L, "SH": VL},
             'MH': {"SL": EH, "SM": VH, "SH": H}}
gamma = 1.5


def get_fuzzy_weight(mu, std):
    if mu <= 0.09:
        a = 'ML'
    elif 0.09 < mu <= 0.22:
        a = 'MM'
    else:
        a = 'MH'
    if std <= 0.03:
        b = 'SL'
    elif 0.03 < std <= 0.12:
        b = 'SM'
    else:
        b = 'SH'
    return omega_dic[a][b]


def can_allocate(workload, capacity):
    for i in range(len(workload)):
        if capacity[i] < workload[i]:
            return False
    return True


def fuzzy_allocate(servers, users, user_masks):
    user_num = len(users)
    server_num = len(servers)
    user_within_servers = mask_trans_to_list(user_masks, server_num)
    # 每个用户被分配到的服务器
    user_allocate_list = [-1] * user_num
    # 每个服务器分配到的用户数量
    server_allocate_num = [0] * server_num

    def allocate(allocated_user_id, allocated_server_id):
        user_allocate_list[allocated_user_id] = allocated_server_id
        server_allocate_num[allocated_server_id] += 1

    # 复制一份server，防止改变工作负载源数据
    tmp_server_capacity = np.array(copy.deepcopy([server[3:] for server in servers]))
    # 为每一个用户分配一个服务器
    for user_id in range(user_num):
        user = users[user_id]
        workload = user[2:]
        # 计算所有服务器的资源利用率
        capacity_used_props = np.zeros(server_num)
        for server_id in range(server_num):
            prop = np.zeros(4)
            for i in range(4):
                prop[i] = 1 - tmp_server_capacity[server_id][i] / servers[server_id][i + 3]
            capacity_used_props[server_id] = np.mean(prop)
        # 计算所有服务器的资源利用率平均值和标准差
        mu = np.mean(capacity_used_props)
        std = np.std(capacity_used_props)

        # 开始遍历服务器找最高分
        final_server_ids = []
        C = []
        B = []
        for server_id in user_within_servers[user_id]:
            capacity = tmp_server_capacity[server_id]
            if can_allocate(workload, capacity):
                # 使用模糊控制机制计算得分
                final_server_ids.append(server_id)
                # 首先计算整合分数c
                # 服务器还没开启，那预计释放时间就是0
                zi = 0 if server_allocate_num[server_id] == 0 else 10
                t = 0  # 当前时间为0
                vj = 10  # 需要占用的时间也为10
                c = abs(zi - (t + vj))  # 所以t + vj是新的预计释放时间，也是10
                if zi < t + vj:
                    c = c * gamma
                C.append(c)
                # 上面的代码实现了：如果开启新的服务器，c = 10 * 1.5 = 15, 否则c = 0
                # 然后计算b，b就是这个服务器四个维度的资源利用率的平均值，b越大，服务器压力越大
                b = capacity_used_props[server_id]
                B.append(b)
        if final_server_ids:
            # 然后就要用模糊控制机制得到权重，从而计算分数
            omega_j = get_fuzzy_weight(mu, std)
            # B 和 C 归一化，然后计算S
            max_c, min_c = max(C), min(C)
            max_b, min_b = max(B), min(B)
            S = []
            for i in range(len(C)):
                ci = (C[i] - min_c) / (max_c - min_c) if max_c - min_c != 0 else 0
                bi = (B[i] - min_b) / (max_b - min_b) if max_b - min_b != 0 else 0
                S.append(omega_j * ci + (1 - omega_j) * bi)

            final_server_id = final_server_ids[np.argmin(np.array(S))]
            tmp_server_capacity[final_server_id] -= workload
            allocate(user_id, final_server_id)
            # 先看C和B到底是要干啥：
            # 首先是argmin，是为了最小化这两个指标
            # C越小，代表不用开启新服务器，所以最小化C是不开启新服务器，也就是用户整合
            # B越小，代表这个服务器越空，所以最小化B，是把用户往空的服务器上分配，偏重负载均衡和开启新服务器

            # 模糊控制的原理：
            # mu越小，std越大，都会让omega越大，也就是更偏重C的分数，C就是偏重整合用户，让用户更集中

    # 已分配用户占所有用户的比例
    allocated_user_num = user_num - user_allocate_list.count(-1)
    user_allocated_prop = allocated_user_num / user_num

    # 已使用服务器占所有服务器比例
    used_server_num = server_num - server_allocate_num.count(0)
    server_used_prop = used_server_num / server_num

    # 已使用的服务器的资源利用率
    server_allocate_mat = np.array(server_allocate_num) > 0
    used_original_server = servers[server_allocate_mat]
    original_servers_capacity = used_original_server[:, 3:]
    servers_remain_capacity = tmp_server_capacity[server_allocate_mat]
    sum_all_capacity = np.sum(original_servers_capacity, axis=0)
    sum_remain_capacity = np.sum(servers_remain_capacity, axis=0)
    # 对于每个维度的资源求资源利用率
    every_capacity_remain_props = np.divide(sum_remain_capacity, sum_all_capacity)
    mean_capacity_remain_props = np.mean(every_capacity_remain_props, axis=0)
    capacity_used_prop = 1 - mean_capacity_remain_props

    return None, None, user_allocate_list, server_allocate_num, \
        user_allocated_prop, server_used_prop, capacity_used_prop

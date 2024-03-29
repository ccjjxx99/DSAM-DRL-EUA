import copy
import random
import numpy as np

from util.utils import mask_trans_to_list


def can_allocate(workload, capacity):
    for i in range(len(workload)):
        if capacity[i] < workload[i]:
            return False
    return True


def random_allocate(servers, users, user_masks):
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
        # 在可用server中随机找一个分配
        random_servers = user_within_servers[user_id][:]
        random.shuffle(random_servers)
        for server_id in random_servers:
            capacity = tmp_server_capacity[server_id]
            if can_allocate(workload, capacity):
                tmp_server_capacity[server_id] -= workload
                allocate(user_id, server_id)
                break
        # within_servers = user.within_servers
        # server_id = random.randint(0, len(within_servers) - 1)
        # server = servers[server_id]
        # capacity = server.capacity
        # if can_allocate(workload, capacity):
        #     servers[server_id].allocate(workload)
        #     allocate(user_id, server_id)

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

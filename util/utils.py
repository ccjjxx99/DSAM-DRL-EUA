import copy

import numpy as np


def log_and_print(log_str, log_filename):
    print(log_str)
    with open(log_filename, "a+", encoding='utf8') as ff:
        ff.write(log_str + '\n')


def can_allocate(workload, capacity):
    for i in range(4):
        if capacity[i] < workload[i]:
            return False
    return True


def get_reward(original_servers, users, actions):
    user_num = len(users)
    server_num = len(original_servers)
    # 每个用户被分配到的服务器
    user_allocate_list = [-1] * user_num
    fake_allocate_list = [-1] * user_num
    # 每个服务器分配到的用户数量
    server_allocate_num = [0] * server_num

    def allocate(allocated_user_id, allocated_server_id):
        user_allocate_list[allocated_user_id] = allocated_server_id
        server_allocate_num[allocated_server_id] += 1

    # 复制一份server，防止改变工作负载源数据
    servers = copy.deepcopy(original_servers)
    # 为每一个用户分配一个服务器
    for user_id in range(user_num):
        user = users[user_id]
        workload = user[2:]

        server_id = actions[user_id]
        server = servers[server_id]
        capacity = server[3:]
        if can_allocate(workload, capacity):
            server[3:] -= workload
            allocate(user_id, server_id)
            # print("用户", user_id, "负载", workload, "服务器", final_server_id, "资源剩余", servers[final_server_id][3:])

    # 已分配用户占所有用户的比例
    allocated_user_num = user_num - user_allocate_list.count(-1)
    user_allocated_prop = allocated_user_num / user_num

    # 已使用服务器占所有服务器比例
    used_server_num = server_num - server_allocate_num.count(0)
    server_used_prop = used_server_num / server_num

    # 已使用的服务器的资源利用率
    server_allocate_mat = np.array(server_allocate_num) > 0
    used_original_server = original_servers[server_allocate_mat]
    original_servers_capacity = used_original_server[:, 3:]
    servers_remain = servers[server_allocate_mat]
    servers_remain_capacity = servers_remain[:, 3:]
    sum_all_capacity = original_servers_capacity.sum()
    sum_remain_capacity = servers_remain_capacity.sum()
    capacity_used_prop = 1 - sum_remain_capacity / sum_all_capacity

    return user_allocate_list, server_allocate_num, user_allocated_prop, server_used_prop, capacity_used_prop

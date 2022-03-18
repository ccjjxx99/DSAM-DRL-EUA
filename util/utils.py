import copy
import logging
from datetime import datetime

import numpy as np
import torch
from numpy import mean
from torch.utils.data import DataLoader
from tqdm import trange, tqdm

from util.EUA_MCF import EUA_MCF


def log_and_print(log_str, log_filename):
    print(log_str)
    try:
        with open(log_filename, "a+", encoding='utf8') as ff:
            ff.write(log_str + '\n')
    except OSError as e:
        print(e)
    except Exception as e:
        print(e)


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


def calc_mcf_reward_by_test_set(test_set):
    servers = test_set.servers
    users_list, users_masks_list = test_set.users_list, test_set.users_masks_list

    user_props = []
    server_props = []
    capacity_props = []
    for i in trange(len(test_set)):
        _, _, _, _, user_allocated_prop, server_used_prop, capacity_prop = EUA_MCF(servers, users_list[i],
                                                                                   users_masks_list[i])
        user_props.append(user_allocated_prop)
        server_props.append(server_used_prop)
        capacity_props.append(capacity_prop)

    return mean(user_props), mean(server_props), mean(capacity_props)


def test_by_model_and_set(model, batch_size, test_set, device):
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)
    # Test
    model.eval()
    model.policy = 'greedy'
    with torch.no_grad():
        test_R_list = []
        test_user_allocated_props_list = []
        test_server_used_props_list = []
        test_capacity_used_props_list = []
        for server_seq, user_seq, masks in tqdm(test_loader):
            server_seq, user_seq, masks = server_seq.to(device), user_seq.to(device), masks.to(device)

            reward, _, action_idx, user_allocated_props, \
                server_used_props, capacity_used_props, user_allocate_list \
                = model(user_seq, server_seq, masks)

            test_R_list.append(reward)
            test_user_allocated_props_list.append(user_allocated_props)
            test_server_used_props_list.append(server_used_props)
            test_capacity_used_props_list.append(capacity_used_props)

        test_R_list = torch.cat(test_R_list)
        test_user_allocated_props_list = torch.cat(test_user_allocated_props_list)
        test_server_used_props_list = torch.cat(test_server_used_props_list)
        test_capacity_used_props_list = torch.cat(test_capacity_used_props_list)

        test_r = torch.mean(test_R_list)
        test_user_allo = torch.mean(test_user_allocated_props_list)
        test_server_use = torch.mean(test_server_used_props_list)
        test_capacity_use = torch.mean(test_capacity_used_props_list)
        print('Ptr:\t', test_user_allo.item(), test_server_use.item(), test_capacity_use.item())


def mask_trans_to_list(user_masks, server_num):
    x = []
    server_arrange = np.arange(server_num)
    for i in range(len(user_masks)):
        mask = user_masks[i]
        y = server_arrange[mask]
        x.append(y.tolist())
    return x
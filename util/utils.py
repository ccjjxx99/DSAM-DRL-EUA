import copy
import logging
import sys

import numpy as np
from numpy import mean
from tqdm import trange


def get_reward(original_servers, users, actions):
    user_num = len(users)
    server_num = len(original_servers)
    # 每个用户被分配到的服务器
    user_allocate_list = [-1] * user_num
    # 每个服务器分配到的用户数量
    server_allocate_num = [0] * server_num

    def can_allocate(user1, server1):
        if not np.linalg.norm(user[:2] - server[:2]) <= server[2]:
            return False
        for i in range(4):
            if user1[i + 2] > server1[i + 3]:
                return False
        return True

    def allocate(allocated_user_id, allocated_server_id):
        user_allocate_list[allocated_user_id] = allocated_server_id
        server_allocate_num[allocated_server_id] += 1

    # 复制一份server，防止改变工作负载源数据
    servers = copy.deepcopy(original_servers)
    # 为每一个用户分配一个服务器
    for user_id in range(user_num):
        user = users[user_id]
        workload = user[2:]

        if actions[user_id] == -1:
            continue
        server_id = actions[user_id]
        server = servers[server_id]
        if can_allocate(user, server):
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
    sum_all_capacity = np.sum(original_servers_capacity, axis=0)
    sum_remain_capacity = np.sum(servers_remain_capacity, axis=0)
    # 对于每个维度的资源求资源利用率
    every_capacity_remain_props = np.divide(sum_remain_capacity, sum_all_capacity)
    mean_capacity_remain_props = np.mean(every_capacity_remain_props, axis=0)
    capacity_used_prop = 1 - mean_capacity_remain_props

    return user_allocate_list, server_allocate_num, user_allocated_prop, server_used_prop, capacity_used_prop


def calc_method_reward_by_test_set(test_set, method, nums=0):
    servers = test_set.servers
    if nums == 0:
        users_list, users_masks_list = test_set.users_list, test_set.users_masks_list
    else:
        users_list, users_masks_list = test_set.users_list[:nums], test_set.users_masks_list[:nums]

    user_props = []
    server_props = []
    capacity_props = []
    for i in trange(len(users_list)):
        _, _, _, _, user_allocated_prop, server_used_prop, capacity_prop = method(servers, users_list[i],
                                                                                  users_masks_list[i])
        user_props.append(user_allocated_prop)
        server_props.append(server_used_prop)
        capacity_props.append(capacity_prop)

    return mean(user_props), mean(server_props), mean(capacity_props)  # , user_props, server_props, capacity_props


def in_coverage(user, server):
    return np.linalg.norm(user[:2] - server[:2]) <= server[2]


def calc_masks_by_test_set(test_set):
    server_list = test_set.servers
    users_list, users_masks_list = test_set.users_list, test_set.users_masks_list

    for k in trange(len(test_set)):
        user_list = users_list[k]
        users_masks = np.zeros((len(user_list), len(server_list)), dtype=np.bool)

        def calc_user_within(calc_user, index):
            flag = False
            for j in range(len(server_list)):
                if in_coverage(calc_user, server_list[j]):
                    users_masks[index, j] = 1
                    flag = True
            return flag

        for i in range(len(user_list)):
            user = user_list[i]
            user_within = calc_user_within(user, i)
            if not user_within:
                print("出大问题")
        if not (users_masks == users_masks_list[k]).all():
            print("出大问题")


def mask_trans_to_list(user_masks, server_num):
    x = []
    user_masks = user_masks.astype(np.bool)
    server_arrange = np.arange(server_num)
    for i in range(len(user_masks)):
        mask = user_masks[i]
        y = server_arrange[mask]
        x.append(y.tolist())
    return x


def save_dataset(save_filename, **train_data):
    print("开始保存数据集至：", save_filename)
    np.savez_compressed(save_filename, **train_data)
    print("保存成功")


def get_logger(log_filename):
    new_logger = logging.getLogger()
    new_logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s: - %(message)s',
        datefmt='%m-%d %H:%M:%S')

    # 使用FileHandler输出到文件
    fh = logging.FileHandler(log_filename)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    # 使用StreamHandler输出到屏幕
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)

    # 添加两个Handler
    new_logger.addHandler(ch)
    new_logger.addHandler(fh)
    return new_logger

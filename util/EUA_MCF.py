import copy
import numpy as np

from util.utils import mask_trans_to_list


def can_allocate(workload, capacity):
    for i in range(4):
        if capacity[i] < workload[i]:
            return False
    return True


def mcf_allocate(original_servers, original_users, original_masks):
    x = zip(original_users, original_masks)
    x = list(x)
    x = sorted(x, key=lambda u: u[0][2])
    users, user_masks = zip(*x)
    users, user_masks = np.array(users, dtype=np.float), np.array(user_masks, dtype=np.bool)
    user_num = len(users)
    server_num = len(original_servers)
    user_within_servers = mask_trans_to_list(user_masks, server_num)
    # 每个用户被分配到的服务器
    user_allocate_list = [-1] * user_num
    fake_allocate_list = [-1] * user_num
    # 每个服务器分配到的用户数量
    server_allocate_num = [0] * server_num

    def allocate(allocated_user_id, allocated_server_id):
        user_allocate_list[allocated_user_id] = allocated_server_id
        fake_allocate_list[allocated_user_id] = allocated_server_id
        server_allocate_num[allocated_server_id] += 1
        activated_servers.append(allocated_server_id)

    # 复制一份server，防止改变工作负载源数据
    servers = copy.deepcopy(original_servers)
    activated_servers = []
    # 为每一个用户分配一个服务器
    for user_id in range(user_num):
        user = users[user_id]
        workload = user[2:]
        # 先过滤为已经激活的
        this_user_s_active_server = []
        other_servers = []
        for server_id in user_within_servers[user_id]:
            if server_id in activated_servers:
                this_user_s_active_server.append(server_id)
            else:
                other_servers.append(server_id)

        # 在可用server中寻找剩余容量最多的
        max_remain_capacity = -1
        final_server_id = -1
        for server_id in this_user_s_active_server:
            server = servers[server_id]
            capacity = server[3:]
            if can_allocate(workload, capacity):
                # 计算总剩余容量
                remain_capacity = sum(capacity) - sum(workload)
                if remain_capacity > max_remain_capacity:
                    max_remain_capacity = remain_capacity
                    final_server_id = server_id
        if final_server_id != -1:
            servers[final_server_id][3:] -= workload
            allocate(user_id, final_server_id)
            # print("用户", user_id, "负载", workload, "服务器", final_server_id, "资源剩余", servers[final_server_id][3:])
        else:
            for server_id in other_servers:
                server = servers[server_id]
                capacity = server[3:]
                if can_allocate(workload, capacity):
                    # 计算总剩余容量
                    remain_capacity = sum(capacity) - sum(workload)
                    if remain_capacity > max_remain_capacity:
                        max_remain_capacity = remain_capacity
                        final_server_id = server_id
            if final_server_id != -1:
                servers[final_server_id][3:] -= workload
                allocate(user_id, final_server_id)
                # print("用户", user_id, "负载", workload, "服务器", final_server_id, "资源剩余", servers[final_server_id][3:])
            else:
                # 如果是-1说明没有服务器装得下它，搞一个假的测试批量reward
                fake_allocate_list[user_id] = user_within_servers[user_id][0]
                # print("用户", user_id, "负载", workload, "无法分配服务器")
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

    return users, fake_allocate_list, user_allocate_list, server_allocate_num, \
        user_allocated_prop, server_used_prop, capacity_used_prop

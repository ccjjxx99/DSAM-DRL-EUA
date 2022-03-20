import math
import random
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from tqdm import tqdm


workload_list = [
    np.array([1, 2, 1, 2]),
    np.array([2, 3, 3, 4]),
    np.array([5, 7, 6, 6])
]


def random_user_load():
    return random.choice(workload_list)


def in_coverage(user, server):
    return np.linalg.norm(user[:2] - server[:2]) <= server[2]


def get_within_servers(user_list, server_list, x_start, x_end, y_start, y_end):
    """
    获取用户被哪些服务器覆盖，如果没有覆盖，则重新生成一个用户
    :return: 重新生成的用户列表；用户覆盖的服务器id
    """
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
        while not user_within:
            user[0] = random.random() * (x_end - x_start) + x_start
            user[1] = random.random() * (y_end - y_start) + y_start
            user_within = calc_user_within(user, i)
    return user_list, users_masks


def get_whole_capacity(user_list, rate):
    """
    获取所有用户的总需求再乘比例
    :param user_list: 用户列表
    :param rate: 资源冗余比例
    :return: 需要的总资源
    """
    capacity = np.sum(user_list[:, 2:], axis=0) * rate
    return capacity


def evaluate_whole_capacity_by_user_num(user_num, rate=3):
    """
    根据用户数量预估总容量
    :param user_num:
    :param rate:
    :return:
    """
    loads = np.array(workload_list)
    average_load = loads.mean(axis=0)
    whole_load = average_load * user_num * rate
    return whole_load


# 为每个服务器分配capacity
def allocate_capacity(server_list, capacity):
    server_len = len(server_list)
    # 对服务器的平均资源添加一些随机因素
    # 最小是0.75倍，最大是1.25倍
    cpu_max = int(capacity[0] * 1.25 / server_len)
    cpu_min = int(capacity[0] * 0.75 / server_len)
    io_max = int(capacity[1] * 1.25 / server_len)
    io_min = int(capacity[1] * 0.75 / server_len)
    bandwidth_max = int(capacity[2] * 1.25 / server_len)
    bandwidth_min = int(capacity[2] / 2 / server_len)
    memory_max = int(capacity[3] * 1.25 / server_len)
    memory_min = int(capacity[3] / 2 / server_len)
    for server in server_list:
        server[3] = random.randint(cpu_min, cpu_max)
        server[4] = random.randint(io_min, io_max)
        server[5] = random.randint(bandwidth_min, bandwidth_max)
        server[6] = random.randint(memory_min, memory_max)
    return server_list


def draw_data(server_list, user_list):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for user in user_list:
        ax.plot(user[0], user[1], 'ro')
    for server in server_list:
        circle = Circle((server[0], server[1]), server[2], alpha=0.2)
        ax.add_patch(circle)
        ax.plot(server[0], server[1], 'bo')
    plt.axis('scaled')
    plt.axis('equal')
    plt.show()


def cal_props_by_seqs(user_seqs, server_seqs, user_allocated_servers):
    batch_size = user_seqs.shape[0]
    user_allocated_props = []
    server_used_props = []
    for i in range(batch_size):
        user_seq = user_seqs[i]
        server_seq = server_seqs[i]
        allocated_seq = user_allocated_servers[i]
        user_allocated_prop, server_used_prop = cal_props(user_seq, server_seq, allocated_seq)
        user_allocated_props.append(user_allocated_prop)
        server_used_props.append(server_used_prop)
    return user_allocated_props, server_used_props


def can_allocate(workload, capacity):
    for i in range(4):
        if capacity[i] < workload[i]:
            return False
    return True


def cal_props(user_seqs, server_seqs, allocated_seq):
    tmp_server_capacity = [server_seq[3:] for server_seq in server_seqs]
    user_num = len(user_seqs)
    server_num = len(server_seqs)
    # 真实分配情况
    user_allocate_list = [-1] * user_num
    server_allocate_num = [0] * server_num

    for i in range(user_num):
        user_seq = user_seqs[i]
        server_id = allocated_seq[i]
        if server_id == -1:
            continue

        if in_coverage(user_seq[:2], server_seqs[server_id][:3]) and can_allocate(user_seq[2:],
                                                                                  tmp_server_capacity[server_id]):
            user_allocate_list[i] = server_id
            server_allocate_num[server_id] += 1
            for j in range(4):
                tmp_server_capacity[server_id][j] -= user_seq[2 + j]

    # 已分配用户占所有用户的比例
    allocated_user_num = user_num - user_allocate_list.count(-1)
    user_allocated_prop = allocated_user_num / user_num

    # 已使用服务器占所有服务器比例
    used_server_num = server_num - server_allocate_num.count(0)
    server_used_prop = used_server_num / server_num

    return user_allocated_prop, server_used_prop


# 获取所有服务器
def get_all_server_xy():
    server_list = []
    file = open("data/site-optus-melbCBD.csv", 'r')
    file.readline().strip()  # 数据集的第一行是字段说明信息，不能作为数据，因此跳过
    lines = file.readlines()
    for i in range(len(lines)):
        result = lines[i].split(',')
        # longitude, latitude
        server_mes = (float(result[2]), float(result[1]))
        x, y = miller_to_xy(*server_mes)
        server_list.append([x, y])
    file.close()

    server_list = np.array(server_list)
    min_xy = np.min(server_list, axis=0)
    server_list -= min_xy

    angel = 13
    for xy in server_list:
        x = xy[0] * math.cos(math.pi / 180 * angel) - xy[1] * math.sin(math.pi / 180 * angel)
        y = xy[0] * math.sin(math.pi / 180 * angel) + xy[1] * math.cos(math.pi / 180 * angel)
        xy[0] = x
        xy[1] = y

    min_xy = np.min(server_list, axis=0)
    server_list -= min_xy

    for xy in server_list:
        xy[0] = xy[0] - xy[1] * math.tan(math.pi / 180 * 15)

    min_xy = np.min(server_list, axis=0)
    server_list -= min_xy
    # 单位：米转换为100米
    server_list /= 100

    return server_list


def miller_to_xy(lon, lat):
    """
    :param lon: 经度
    :param lat: 维度
    :return:
    """
    L = 6381372 * math.pi * 2  # 地球周长
    W = L  # 平面展开，将周长视为X轴
    H = L / 2  # Y轴约等于周长一半
    mill = 2.3  # 米勒投影中的一个常数，范围大约在正负2.3之间
    x = lon * math.pi / 180  # 将经度从度数转换为弧度
    y = lat * math.pi / 180

    y = 1.25 * math.log(math.tan(0.25 * math.pi + 0.4 * y))  # 这里是米勒投影的转换

    # 这里将弧度转为实际距离
    x = (W / 2) + (W / (2 * math.pi)) * x
    y = (H / 2) - (H / (2 * mill)) * y
    return x, y


def init_server(x_start_prop, x_end_prop, y_start_prop, y_end_prop,
                min_cov=1, max_cov=1.5, miu=35, sigma=10):
    """
    根据比例从地图中截取一些服务器的坐标
    """
    server_xy_list = get_all_server_xy()
    max_x_y = np.max(server_xy_list, axis=0)
    max_x = max_x_y[0]
    max_y = max_x_y[1]
    x_start = max_x * x_start_prop
    x_end = max_x * x_end_prop
    y_start = max_y * y_start_prop
    y_end = max_y * y_end_prop
    filter_server = [x_start <= server[0] <= x_end
                     and y_start <= server[1] <= y_end
                     for server in server_xy_list]
    server_xy_list = server_xy_list[filter_server]
    # 将这些服务器最左上角定义为(0,0)+覆盖范围
    min_xy = np.min(server_xy_list, axis=0)
    server_xy_list = server_xy_list - min_xy + max_cov
    server_cov_list = np.random.uniform(min_cov, max_cov, (len(server_xy_list), 1))
    server_capacity_list = np.random.normal(miu, sigma, size=(len(server_xy_list), 4))
    server_list = np.concatenate((server_xy_list, server_cov_list, server_capacity_list), axis=1)
    return server_list


def init_users_list_by_server(server_list, data_num, user_num, load_sorted=True, max_cov=1.5):
    """
    固定服务器坐标，生成一组user，同时补充服务器的资源容量
    :param server_list:
    :param data_num: 生成多少组
    :param user_num: 用户数
    :param max_cov: 最大覆盖半径，给左上角坐标加上，以免用户只能在左上角第一个服务器的右下角1/4的范围内生成
    :param load_sorted: 是否直接生成已按load排序的用户
    :return:
    """
    max_server = np.max(server_list, axis=0)
    max_x = max_server[0] + max_cov
    max_y = max_server[1] + max_cov
    min_server = np.min(server_list, axis=0)
    min_x = min_server[0] - max_cov
    min_y = min_server[1] - max_cov

    users_list = []
    users_masks_list = []
    for _ in tqdm(range(data_num)):
        user_x_list = np.random.uniform(min_x, max_x, (user_num, 1))
        user_y_list = np.random.uniform(min_y, max_y, (user_num, 1))
        if load_sorted:
            num01 = int(1 / 3 * user_num)
            num2 = user_num - 2 * num01
            w0 = np.tile(workload_list[0], (num01, 1))
            w1 = np.tile(workload_list[1], (num01, 1))
            w2 = np.tile(workload_list[2], (num2, 1))
            user_load_list = np.concatenate((w0, w1, w2), axis=0)
        else:
            user_load_list = np.array([random_user_load() for _ in range(user_num)])
        user_list = np.concatenate((user_x_list, user_y_list, user_load_list), axis=1)
        user_list, users_masks = get_within_servers(user_list, server_list, min_x, max_x, min_y, max_y)
        users_list.append(user_list)
        users_masks_list.append(users_masks)

    return {"users_list": users_list, "users_masks_list": users_masks_list}

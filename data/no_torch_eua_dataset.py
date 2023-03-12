import os
import numpy as np


class EuaDataset:
    def __init__(self, servers, users_list, users_masks_list, device):
        self.servers, self.users_list, self.users_masks_list = servers, users_list, users_masks_list
        self.device = device

    def __len__(self):
        return len(self.users_list)


class EuaDatasetNeedSort:
    def __init__(self, servers, users_list, users_masks_list, device):
        self.servers, self.users_list, self.users_masks_list = servers, users_list, users_masks_list
        self.device = device

    def __len__(self):
        return len(self.users_list)


def get_dataset(x_end, y_end, miu, sigma, user_num, data_size: {}, min_cov, max_cov, device, dir_name):
    """
    获取dataset
    :param x_end:
    :param y_end:
    :param miu:
    :param sigma:
    :param user_num:
    :param data_size: 字典，key为dataset类型，value为该类型的数量
    :param min_cov:
    :param max_cov:
    :param device:
    :param dir_name: 数据集存放的文件夹
    :return:
    """
    dataset_dir_name = os.path.join(dir_name,
                                    "dataset/server_" + str(x_end) + "_" + str(y_end)
                                    + "_miu_" + str(miu) + "_sigma_" + str(sigma))
    server_file_name = "server_" + str(x_end) + "_" + str(y_end) + "_miu_" + str(miu) + "_sigma_" + str(sigma)
    server_path = os.path.join(dataset_dir_name, server_file_name) + '.npy'
    if os.path.exists(server_path):
        servers = np.load(server_path)
        print("读取服务器数据成功")
    else:
        print("未读取到服务器数据，错误")
        exit(-1)
    set_types = data_size.keys()
    datasets = {}
    for set_type in set_types:
        if set_type not in ('train', 'valid', 'test'):
            raise NotImplementedError
        filename = set_type + "_user_" + str(user_num) + "_size_" + str(data_size[set_type])
        path = os.path.join(dataset_dir_name, filename) + '.npz'
        if os.path.exists(path):
            print("正在加载", set_type, "数据集")
            data = np.load(path)
        else:
            print(set_type, "数据集未找到，错误", path)
            exit(-1)
        datasets[set_type] = EuaDataset(servers, **data, device=device)

    return datasets


def shuffle_dataset(test_set):
    new_users = []
    new_masks = []
    for i in range(len(test_set)):
        x = zip(test_set.users_list[i], test_set.users_masks_list[i])
        x = list(x)
        np.random.shuffle(x)
        new_user, new_mask = zip(*x)
        new_users.append(new_user)
        new_masks.append(new_mask)
    new_users_array = np.stack(new_users)
    new_masks_array = np.stack(new_masks)
    return EuaDataset(test_set.servers, new_users_array, new_masks_array, test_set.device)

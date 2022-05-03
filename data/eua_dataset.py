import os
import numpy as np
import torch
from torch.utils.data import Dataset

from data.data_generator import init_server, init_users_list_by_server
from util.utils import save_dataset


class EuaTrainDataset(Dataset):
    def __init__(self, servers, users_list, users_within_servers_list, users_masks_list, device):
        self.servers = torch.tensor(servers, dtype=torch.float32, device=device)
        self.users_list, self.users_within_servers_list, self.users_masks_list = \
            users_list, users_within_servers_list, users_masks_list
        self.device = device

    def __len__(self):
        return len(self.users_list)

    def __getitem__(self, index):
        user_seq = torch.tensor(self.users_list[index], dtype=torch.float32, device=self.device)
        mask_seq = torch.tensor(self.users_masks_list[index], dtype=torch.bool, device=self.device)
        return self.servers, user_seq, mask_seq


class EuaDataset(Dataset):
    def __init__(self, servers, users_list, users_masks_list, device):
        self.servers, self.users_list, self.users_masks_list = servers, users_list, users_masks_list
        self.servers_tensor = torch.tensor(servers, dtype=torch.float32, device=device)
        self.device = device

    def __len__(self):
        return len(self.users_list)

    def __getitem__(self, index):
        user_seq = torch.tensor(self.users_list[index], dtype=torch.float32, device=self.device)
        mask_seq = torch.tensor(self.users_masks_list[index], dtype=torch.bool, device=self.device)
        return self.servers_tensor, user_seq, mask_seq


class EuaDatasetNeedSort(Dataset):
    def __init__(self, servers, users_list, users_masks_list, device):
        self.servers, self.users_list, self.users_masks_list = servers, users_list, users_masks_list
        self.servers_tensor = torch.tensor(servers, dtype=torch.float32, device=device)
        self.device = device

    def __len__(self):
        return len(self.users_list)

    def __getitem__(self, index):
        # 先排序
        original_users = self.users_list[index]
        users = sorted(original_users, key=lambda u: u[2])
        user_seq = torch.tensor(original_users, dtype=torch.float32, device=self.device)
        mask_seq = torch.tensor(self.users_masks_list[index], dtype=torch.bool, device=self.device)
        return self.servers_tensor, user_seq, mask_seq


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
        print("未读取到服务器数据，重新生成")
        os.makedirs(dataset_dir_name, exist_ok=True)
        servers = init_server(0, x_end, 0, y_end, min_cov, max_cov, miu, sigma)
        np.save(server_path, servers)
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
            print(set_type, "数据集未找到，重新生成", path)
            data = init_users_list_by_server(servers, data_size[set_type], user_num, True, max_cov)
            save_dataset(path, **data)
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

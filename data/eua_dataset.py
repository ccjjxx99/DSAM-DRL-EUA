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


def get_dataset(x_end, y_end, miu, sigma, user_num, data_size, min_cov, max_cov, device):
    dataset_dir_name = "D:/transformer_eua/dataset/server_" + str(x_end) + "_" + str(y_end) \
                       + "_miu_" + str(miu) + "_sigma_" + str(sigma)
    server_file_name = "server_" + str(x_end) + "_" + str(y_end) + "_miu_" + str(miu) + "_sigma_" + str(sigma)
    server_path = os.path.join(dataset_dir_name, server_file_name) + '.npy'

    train_filename = "train_user_" + str(user_num) + "_size_" + str(data_size['train'])
    valid_filename = "valid_user_" + str(user_num) + "_size_" + str(data_size['valid'])
    test_filename = "test_user_" + str(user_num) + "_size_" + str(data_size['test'])

    path = {'train': os.path.join(dataset_dir_name, train_filename) + '.npz',
            'valid': os.path.join(dataset_dir_name, valid_filename) + '.npz',
            'test': os.path.join(dataset_dir_name, test_filename) + '.npz'}
    set_types = ['train', 'valid', 'test']
    # 判断目录是否存在
    if os.path.exists(server_path):
        servers = np.load(server_path)
        print("读取服务器数据成功")
    else:
        print("未读取到服务器数据，重新生成")
        os.makedirs(dataset_dir_name, exist_ok=True)
        servers = init_server(0, x_end, 0, y_end, min_cov, max_cov, miu, sigma)
        np.save(server_path, servers)
    datas = []
    for set_type in set_types:
        if os.path.exists(path[set_type]):
            print("正在加载", set_type, "数据集")
            data = np.load(path[set_type])
            datas.append(data)
        else:
            print(set_type, "数据集未找到，重新生成")
            data = init_users_list_by_server(servers, data_size[set_type], user_num, True, max_cov)
            datas.append(data)
            save_dataset(path[set_type], **data)
    train_set = EuaDataset(servers, **datas[0], device=device)
    valid_set = EuaDataset(servers, **datas[1], device=device)
    test_set = EuaDataset(servers, **datas[2], device=device)

    return {'train': train_set, 'valid': valid_set, 'test': test_set}

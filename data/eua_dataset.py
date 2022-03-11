import torch
from torch.utils.data import Dataset
from data.data_generator import DataGenerator


class EuaDataset(Dataset):
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


def generate_three_set(user_num, data_num, x_start_prop, x_end_prop, y_start_prop, y_end_prop, device,
                       min_cov=1, max_cov=1.5, miu=35, sigma=10):
    """
    :param user_num:
    :param data_num: 数组，包括训练、验证、测试三个数
    :param x_start_prop:
    :param x_end_prop:
    :param y_start_prop:
    :param y_end_prop:
    :param device:
    :param min_cov:
    :param max_cov:
    :param miu: 服务器容量的均值
    :param sigma: 服务器容量的方差
    :return:
    """
    generator = DataGenerator()
    servers = generator.init_server(x_start_prop, x_end_prop, y_start_prop, y_end_prop, min_cov, max_cov)
    users_list, users_within_servers_list, users_masks_list = \
        generator.init_users_list_by_server(servers, data_num[0], user_num, load_sorted=True, max_cov=max_cov)
    train_set = EuaDataset(servers, users_list, users_within_servers_list, users_masks_list, device)

    users_list, users_within_servers_list, users_masks_list = \
        generator.init_users_list_by_server(servers, data_num[1], user_num, load_sorted=True, max_cov=max_cov)
    valid_set = EuaDataset(servers, users_list, users_within_servers_list, users_masks_list, device)

    users_list, users_within_servers_list, users_masks_list = \
        generator.init_users_list_by_server(servers, data_num[2], user_num, load_sorted=True, max_cov=max_cov)
    test_set = EuaDataset(servers, users_list, users_within_servers_list, users_masks_list, device)

    return train_set, valid_set, test_set

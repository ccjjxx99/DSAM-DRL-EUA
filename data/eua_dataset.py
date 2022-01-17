import torch
from torch.utils.data import Dataset
from data.data_generator import DataGenerator


class EuaTrainDataset(Dataset):

    def __init__(self, user_num, data_num, x_start_prop, x_end_prop, y_start_prop, y_end_prop, device,
                 rate=3, min_cov=1, max_cov=1.5):
        self.generator = DataGenerator(user_num, rate)
        servers = self.generator.init_server(x_start_prop, x_end_prop, y_start_prop, y_end_prop, min_cov, max_cov)
        self.servers = torch.tensor(servers, dtype=torch.float32, device=device)
        self.users_list, self.users_within_servers_list, self.users_masks_list =\
            self.generator.init_users_list_by_server(servers, data_num, True, max_cov)
        self.device = device

    def __len__(self):
        return len(self.users_list)

    def __getitem__(self, index):
        user_seq = torch.tensor(self.users_list[index], dtype=torch.float32, device=self.device)
        mask_seq = torch.tensor(self.users_masks_list[index], dtype=torch.bool, device=self.device)
        return self.servers, user_seq, mask_seq


class EuaTestDataset(Dataset):

    def __init__(self, user_num, data_num, x_start_prop, x_end_prop, y_start_prop, y_end_prop, device,
                 rate=3, min_cov=1, max_cov=1.5):
        self.generator = DataGenerator(user_num, rate)
        servers = self.generator.init_server(x_start_prop, x_end_prop, y_start_prop, y_end_prop, min_cov, max_cov)
        self.servers = torch.tensor(servers, device=device)
        self.users_list, self.users_within_servers_list, self.users_masks_list =\
            self.generator.init_users_list_by_server(self.servers, data_num, False, max_cov)
        self.device = device

    def __len__(self):
        return len(self.users_list)

    def __getitem__(self, index):
        user_seq = self.users_list[index]
        mask_seq = self.users_masks_list[index]

        zipped = zip(user_seq, mask_seq)
        # 按照CPU使用量排序
        sort_zipped = sorted(zipped, key=lambda x: (x[0][2]))

        result = zip(*sort_zipped)
        user_seq, mask_seq = [list(x) for x in result]
        user_seq = torch.tensor(user_seq, device=self.device)
        mask_seq = torch.tensor(mask_seq, device=self.device)

        return self.servers, user_seq, mask_seq

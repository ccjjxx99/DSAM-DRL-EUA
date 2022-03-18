import torch
from torch.utils.data import Dataset


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

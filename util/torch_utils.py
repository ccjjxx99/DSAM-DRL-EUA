import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from util.utils import get_reward


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

            reward, _, action_idx, user_allocated_props, server_used_props, capacity_used_props, user_allocate_list \
                = model(user_seq, server_seq, masks)

            # 将batch_size设为1并取消注释下面的代码，可以测试模型自带的reward函数计算是否准确
            # print(user_allocated_props[0].item(), server_used_props[0].item(), capacity_used_props[0].item())
            #
            # _, _, user_allocated_prop, server_used_prop, capacity_used_prop \
            #     = get_reward(server_seq.squeeze(0).cpu().numpy(),
            #                  user_seq.squeeze(0).cpu().numpy(),
            #                  user_allocate_list.squeeze(0).cpu().numpy())
            # print(user_allocated_prop, server_used_prop, capacity_used_prop)

            test_R_list.append(reward)
            test_user_allocated_props_list.append(user_allocated_props)
            test_server_used_props_list.append(server_used_props)
            test_capacity_used_props_list.append(capacity_used_props)

        # test_R_list = torch.cat(test_R_list)
        test_user_allocated_props_list = torch.cat(test_user_allocated_props_list)
        test_server_used_props_list = torch.cat(test_server_used_props_list)
        test_capacity_used_props_list = torch.cat(test_capacity_used_props_list)

        # test_r = torch.mean(test_R_list)
        test_user_allo = torch.mean(test_user_allocated_props_list)
        test_server_use = torch.mean(test_server_used_props_list)
        test_capacity_use = torch.mean(test_capacity_used_props_list)
        print('AM-DRL:\t', test_user_allo.item(), test_server_use.item(), test_capacity_use.item())


def test_by_model_and_set_without_batch(model, test_set, device):
    test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False)
    # Test
    model.eval()
    model.policy = 'greedy'
    with torch.no_grad():
        # test_R_list = []
        test_user_allocated_props_list = []
        test_server_used_props_list = []
        test_capacity_used_props_list = []
        for server_seq, user_seq, masks in tqdm(test_loader):
            server_seq, user_seq, masks = server_seq.to(device), user_seq.to(device), masks.to(device)

            _, _, action_idx, _, _, _, user_allocate_list = model(user_seq, server_seq, masks)

            _, _, user_allocated_prop, server_used_prop, capacity_used_prop \
                = get_reward(server_seq.squeeze(0).cpu().numpy(),
                             user_seq.squeeze(0).cpu().numpy(),
                             user_allocate_list.squeeze(0).cpu().numpy())

            # test_R_list.append(reward)
            test_user_allocated_props_list.append(user_allocated_prop)
            test_server_used_props_list.append(server_used_prop)
            test_capacity_used_props_list.append(capacity_used_prop)

        # test_r = torch.mean(test_R_list)
        test_user_allo = np.mean(test_user_allocated_props_list)
        test_server_use = np.mean(test_server_used_props_list)
        test_capacity_use = np.mean(test_capacity_used_props_list)
        print('AM-DRL:\t', test_user_allo.item(), test_server_use.item(), test_capacity_use.item())


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

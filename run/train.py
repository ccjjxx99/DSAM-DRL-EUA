import pickle
import time
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from util.eua_dataset import EuaTrainDataset
from nets.attention_net import PointerNet
from util.utils import log_and_print

if __name__ == '__main__':
    batch_size = 1024
    no_cuda = False
    use_cuda = not no_cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    lr = 1e-2
    beta = 0.9
    max_grad_norm = 2.
    epochs = 300
    dropout = 0.5
    server_reward_rate=0.01

    user_num = 100

    train_size = 100000
    valid_size = 10000
    test_size = 10000

    print("正在加载训练数据集")
    filename = "../data/train_server_22_user_" + str(user_num) + "_" + str(train_size) + ".pkl"
    try:
        with open(filename, 'rb') as f:
            train_set = pickle.load(f)  # read file and build object
    except FileNotFoundError:
        train_set = EuaTrainDataset(user_num, train_size, 0, 0.4, 0, 0.5, device)
        with open(filename, 'wb') as f:
            pickle.dump(train_set, f)
    print("加载训练数据集完成")

    filename = "../data/valid_server_22_user_" + str(user_num) + "_" + str(valid_size) + ".pkl"
    try:
        with open(filename, 'rb') as f:
            valid_set = pickle.load(f)  # read file and build object
    except FileNotFoundError:
        valid_set = EuaTrainDataset(user_num, valid_size, 0, 0.4, 0, 0.5, device)
        with open(filename, 'wb') as f:
            pickle.dump(valid_set, f)
    print("加载验证数据集完成")

    filename = "../data/test_server_22_user_" + str(user_num) + "_" + str(test_size) + ".pkl"
    try:
        with open(filename, 'rb') as f:
            test_set = pickle.load(f)  # read file and build object
    except FileNotFoundError:
        test_set = EuaTrainDataset(user_num, test_size, 0, 0.4, 0, 0.5, device)
        with open(filename, 'wb') as f:
            pickle.dump(test_set, f)
    print("加载测试数据集完成")

    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)

    model = PointerNet(6, 7, 256, device=device, dropout=dropout, server_reward_rate=server_reward_rate)
    optimizer = Adam(model.parameters(), lr=lr)

    critic_exp_mvg_avg = torch.zeros(1, device=device)
    log_file_name = "../log/rl_log" + time.strftime('%m%d%H%M', time.localtime(time.time())) + '.log'
    for epoch in range(epochs):
        # Train
        model.train()
        for batch_idx, (server_seq, user_seq, masks) in enumerate(train_loader):
            server_seq, user_seq, masks = server_seq.to(device), user_seq.to(device), masks.to(device)

            optimizer.zero_grad()
            reward, actions_probs, action_idx, user_allocated_props, server_used_props, user_allocate_list \
                = model(user_seq, server_seq, masks)

            if batch_idx == 0:
                critic_exp_mvg_avg = reward.mean()
            else:
                critic_exp_mvg_avg = (critic_exp_mvg_avg * beta) + ((1. - beta) * reward.mean())

            advantage = reward - critic_exp_mvg_avg

            log_probs = torch.zeros(user_seq.size(0), device=device)
            for prob in actions_probs:
                log_prob = torch.log(prob)
                log_probs += log_prob
            log_probs[log_probs < -1000] = -1000.

            reinforce = advantage * log_probs
            actor_loss = reinforce.mean()

            optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(max_grad_norm), norm_type=2)

            optimizer.step()

            critic_exp_mvg_avg = critic_exp_mvg_avg.detach()

            log_and_print(
                '{} Epoch {}: Train [{}/{} ({:.1f}%)]\tR:{:.6f}\tuser_props: {:.6f}\tserver_props: {:.6f}'
                .format(time.strftime('%H:%M:%S', time.localtime(time.time())),
                        epoch, (batch_idx + 1) * len(user_seq), train_size,
                        100. * (batch_idx + 1) / len(train_loader), torch.mean(reward),
                        torch.mean(user_allocated_props),
                        torch.mean(server_used_props)),
                log_file_name)

        # Valid
        model.eval()
        with torch.no_grad():
            for batch_idx, (server_seq, user_seq, masks) in enumerate(valid_loader):
                server_seq, user_seq, masks = server_seq.to(device), user_seq.to(device), masks.to(device)

                reward, actions_probs, action_idx, user_allocated_props, server_used_props, user_allocate_list \
                    = model(user_seq, server_seq, masks)

                log_and_print(
                    '{} Epoch {}: Valid [{}/{} ({:.1f}%)]\tR:{:.6f}\tuser_props: {:.6f}\tserver_props: {:.6f}'
                    .format(time.strftime('%H:%M:%S', time.localtime(time.time())),
                            epoch, (batch_idx + 1) * len(user_seq), valid_size,
                            100. * (batch_idx + 1) / len(valid_loader), torch.mean(reward),
                            torch.mean(user_allocated_props),
                            torch.mean(server_used_props)),
                    log_file_name)

            # Test
            R_list = []
            user_allocated_props_list = []
            server_used_props_list = []
            for batch_idx, (server_seq, user_seq, masks) in enumerate(test_loader):
                server_seq, user_seq, masks = server_seq.to(device), user_seq.to(device), masks.to(device)

                reward, actions_probs, action_idx, user_allocated_props, server_used_props, user_allocate_list \
                    = model(user_seq, server_seq, masks)

                R_list.append(reward)
                user_allocated_props_list.append(user_allocated_props)
                server_used_props_list.append(server_used_props)

            R_list = torch.cat(R_list)
            user_allocated_props_list = torch.cat(user_allocated_props_list)
            server_used_props_list = torch.cat(server_used_props_list)
            log_and_print('{} Epoch {}: Test \tR:{:.6f}\tuser_props: {:.6f}\tserver_props: {:.6f}'
                          .format(time.strftime('%H:%M:%S', time.localtime(time.time())),
                                  epoch, torch.mean(R_list),
                                  torch.mean(user_allocated_props_list),
                                  torch.mean(server_used_props_list)),
                          log_file_name)
            log_and_print('', log_file_name)

            # 每10个epoch保存一次模型：
            if epoch % 10 == 9:
                model_filename = "../model/model_RL_{}_props_{:.6f}.mdl" \
                    .format(time.strftime('%m%d%H%M', time.localtime(time.time())), torch.mean(R_list))
                torch.save(model, model_filename)
                log_and_print("模型已存储到: {}".format(model_filename), log_file_name)

            torch.cuda.empty_cache()

            # if epoch % 20 == 19:
            #     lr = lr / 10
            #     for param_group in optimizer.param_groups:
            #         param_group['lr'] = lr
            #     log_and_print("调整学习率为: {}".format(lr), log_file_name)

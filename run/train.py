import pickle
import time
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from nets.attention_net import PointerNet
from util.utils import log_and_print
from data.eua_dataset import generate_three_set

if __name__ == '__main__':
    batch_size = 2048
    no_cuda = False
    use_cuda = not no_cuda and torch.cuda.is_available()
    lr = 1e-4
    beta = 0.9
    max_grad_norm = 2.
    epochs = 1000
    dropout = 0.1
    server_reward_rate = 0.1
    user_num = 100
    resource_rate = 1.5
    x_end = 0.4
    y_end = 0.5
    train_size = 100000
    valid_size = 10000
    test_size = 10000
    device = torch.device("cuda:0" if use_cuda else "cpu")

    train_filename = "D:/transformer_eua/dataset/train_server_" + str(x_end) + "_" + str(y_end) + "_user_" \
                     + str(user_num) + "_rate_" + str(resource_rate) + "_size_" + str(train_size) + ".pkl"
    valid_filename = "D:/transformer_eua/dataset/valid_server_" + str(x_end) + "_" + str(y_end) + "_user_" \
                     + str(user_num) + "_rate_" + str(resource_rate) + "_size_" + str(valid_size) + ".pkl"
    test_filename = "D:/transformer_eua/dataset/test_server_" + str(x_end) + "_" + str(y_end) + "_user_" \
                    + str(user_num) + "_rate_" + str(resource_rate) + "_size_" + str(test_size) + ".pkl"

    try:
        print("正在加载训练数据集")
        with open(train_filename, 'rb') as f:
            train_set = pickle.load(f)
        print("正在加载验证数据集")
        with open(valid_filename, 'rb') as f:
            valid_set = pickle.load(f)
        print("正在加载测试数据集")
        with open(test_filename, 'rb') as f:
            test_set = pickle.load(f)
    except FileNotFoundError as e:
        print("文件{}未找到，重新生成".format(e.filename))
        train_set, valid_set, test_set = \
            generate_three_set(user_num, (train_size, valid_size, test_size),
                               0, x_end, 0, y_end, device, rate=resource_rate)
        with open(train_filename, 'wb') as f:
            pickle.dump(train_set, f)
            print("保存训练集成功")
        with open(valid_filename, 'wb') as f:
            pickle.dump(valid_set, f)
            print("保存验证集成功")
        with open(test_filename, 'wb') as f:
            pickle.dump(test_set, f)
            print("保存测试集成功")

    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)

    model = PointerNet(6, 7, 256, device=device, dropout=dropout, server_reward_rate=server_reward_rate)
    optimizer = Adam(model.parameters(), lr=lr)

    critic_exp_mvg_avg = torch.zeros(1, device=device)
    log_file_name = "../log/" + time.strftime('%m%d%H%M', time.localtime(time.time())) \
                    + "_server_" + str(x_end) + "_" + str(y_end) + "_user_" \
                    + str(user_num) + "_rate_" + str(resource_rate) + '.log'

    test_reward_list = []
    test_user_list = []
    test_server_list = []
    for epoch in range(epochs):
        # Train
        model.train()
        model.policy = 'sample'
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
                '{} Epoch {}: Train [{}/{} ({:.1f}%)]\tR:{:.6f}\tuser_props: {:.6f}\tserver_props: {:.6f}'.format(
                    time.strftime('%H:%M:%S', time.localtime(time.time())),
                    epoch,
                    (batch_idx + 1) * len(user_seq),
                    train_size,
                    100. * (batch_idx + 1) / len(train_loader),
                    torch.mean(reward),
                    torch.mean(user_allocated_props),
                    torch.mean(server_used_props)),
                log_file_name)

        # Valid
        model.eval()
        log_and_print('', log_file_name)
        with torch.no_grad():
            for batch_idx, (server_seq, user_seq, masks) in enumerate(valid_loader):
                server_seq, user_seq, masks = server_seq.to(device), user_seq.to(device), masks.to(device)

                reward, actions_probs, action_idx, user_allocated_props, server_used_props, user_allocate_list \
                    = model(user_seq, server_seq, masks)

                log_and_print(
                    '{} Epoch {}: Valid [{}/{} ({:.1f}%)]\tR:{:.6f}\tuser_props: {:.6f}\tserver_props: {:.6f}'.format(
                        time.strftime('%H:%M:%S', time.localtime(time.time())),
                        epoch,
                        (batch_idx + 1) * len(user_seq),
                        valid_size,
                        100. * (batch_idx + 1) / len(valid_loader),
                        torch.mean(reward),
                        torch.mean(user_allocated_props),
                        torch.mean(server_used_props)),
                    log_file_name)

            # Test
            R_list = []
            user_allocated_props_list = []
            server_used_props_list = []
            model.policy = 'greedy'
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
            r = torch.mean(R_list)
            user_allo = torch.mean(user_allocated_props_list)
            server_use = torch.mean(server_used_props_list)
            log_and_print('{} Epoch {}: Test \tR:{:.6f}\tuser_props: {:.6f}\tserver_props: {:.6f}'.format(
                time.strftime('%H:%M:%S', time.localtime(time.time())), epoch, r, user_allo, server_use),
                log_file_name)
            log_and_print('', log_file_name)

            test_reward_list.append(r)
            test_user_list.append(user_allo)
            test_server_list.append(server_use)
            if len(test_reward_list) == 1:
                best_r = r
                best_user = user_allo
                best_server = server_use
                best_state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
                best_time = 0
            else:
                if r < best_r:
                    best_r = r
                    best_user = user_allo
                    best_server = server_use
                    best_time = 0
                else:
                    best_time += 1

            # 每20个epoch保存一次模型：
            if epoch % 20 == 19:
                model_filename = "../model/" + time.strftime('%m%d%H%M', time.localtime(time.time())) \
                                 + "_server_" + str(x_end) + "_" + str(y_end) + "_user_" \
                                 + str(user_num) + "_rate_" + str(resource_rate) + '.mdl'
                state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
                torch.save(state, model_filename)
                log_and_print("模型已存储到: {}".format(model_filename), log_file_name)
                # 加载方法
                # checkpoint = torch.load(path)
                # model.load_state_dict(checkpoint['model'])
                # optimizer.load_state_dict(checkpoint['optimizer'])
                # epoch = checkpoint(['epoch'])

            torch.cuda.empty_cache()

            # 如果超过20个epoch奖励都没有再提升，就停止训练
            if best_time >= 20:
                log_and_print("效果如下：", log_file_name)
                for i in range(len(test_reward_list)):
                    log_and_print("Epoch: {}\treward: {}\tuser_props: {}\tserver_props: {}"
                                  .format(i, test_reward_list[i], test_user_list[i], test_server_list[i]),
                                  log_file_name)
                model_filename = "../model/" + time.strftime('%m%d%H%M', time.localtime(time.time())) \
                                 + "_server_" + str(x_end) + "_" + str(y_end) + "_user_" \
                                 + str(user_num) + "_rate_" + str(resource_rate) + "_" \
                                 + "%.2f" % (best_user * 100) + "_" + "%.2f" % (best_server * 100) + '.mdl'
                torch.save(best_state, model_filename)
                log_and_print("模型已存储到: {}".format(model_filename), log_file_name)
                log_and_print("训练结束，最好的reward:{}，用户分配率:{}，服务器租用率:{}"
                              .format(best_r, best_user, best_server), log_file_name)
                exit()

            # if epoch % 20 == 19:
            #     lr = lr / 10
            #     for param_group in optimizer.param_groups:
            #         param_group['lr'] = lr
            #     log_and_print("调整学习率为: {}".format(lr), log_file_name)

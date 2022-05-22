import os
import time
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import yaml

from nets.attention_net import PointerNet, CriticNet
from util.utils import seed_torch, get_logger
from data.eua_dataset import get_dataset


def train(config):
    seed_torch()
    train_config, data_config, model_config = config['train'], config['data'], config['model']
    assert torch.cuda.is_available(), 'cuda无法使用'
    device = train_config['device']
    dataset = get_dataset(data_config['x_end'], data_config['y_end'], data_config['miu'], data_config['sigma'],
                          data_config['user_num'], data_config['data_size'],
                          data_config['min_cov'], data_config['max_cov'], device, train_config['dir_name'])
    train_loader = DataLoader(dataset=dataset['train'], batch_size=train_config['batch_size'], shuffle=True)
    valid_loader = DataLoader(dataset=dataset['valid'], batch_size=train_config['batch_size'], shuffle=False)
    test_loader = DataLoader(dataset=dataset['test'], batch_size=train_config['batch_size'], shuffle=False)

    model = PointerNet(6, 7, 256, device=device,
                       capacity_reward_rate=model_config['capacity_reward_rate'],
                       user_embedding_type=model_config['user_embedding_type'],
                       server_embedding_type=model_config['server_embedding_type'])
    optimizer = Adam(model.parameters(), lr=train_config['lr'])
    exploration_c = train_config['exploration_c']
    original_train_type = train_config['train_type']
    if original_train_type == 'REINFORCE+RGRB':
        now_train_type = 'REINFORCE'
    else:
        now_train_type = original_train_type
    critic_model = None
    critic_optimizer = None
    if now_train_type == 'ac':
        critic_model = CriticNet(6, 7, 256, device, model['user_embedding_type'], model['server_embedding_type'])
        critic_optimizer = Adam(critic_model.parameters(), lr=train_config['lr'])

    # 加载需要继续训练的模型
    if model_config['need_continue']:
        checkpoint = torch.load(model_config['continue_model_filename'], map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1

        if now_train_type == 'ac':
            critic_model.load_state_dict(checkpoint['critic_model'])
            critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])

        print("成功导入预训练模型")
    else:
        start_epoch = 0

    critic_exp_mvg_avg = torch.zeros(1, device=device)

    dir_name = "" + time.strftime('%m%d%H%M', time.localtime(time.time())) \
               + "_server_" + str(data_config['x_end']) + "_" + str(data_config['y_end']) \
               + "_user_" + str(data_config['user_num']) \
               + "_miu_" + str(data_config['miu']) + "_sigma_" + str(data_config['sigma']) \
               + "_" + model_config['user_embedding_type'] + "_" + model_config['server_embedding_type'] \
               + "_" + now_train_type + "_capa_rate_" + str(model_config['capacity_reward_rate'])
    dir_name = os.path.join(train_config['dir_name'], dir_name)
    log_file_name = dir_name + '/log.log'

    os.makedirs(dir_name, exist_ok=True)
    tensorboard_writer = SummaryWriter(dir_name)
    logger = get_logger(log_file_name)
    now_exit = False

    all_valid_reward_list = []
    all_valid_user_list = []
    all_valid_server_list = []
    all_valid_capacity_list = []
    best_r = 0
    for epoch in range(start_epoch, train_config['epochs']):
        # Train
        model.train()
        model.policy = 'sample'
        model.beam_num = 1
        for batch_idx, (server_seq, user_seq, masks) in enumerate(train_loader):
            server_seq, user_seq, masks = server_seq.to(device), user_seq.to(device), masks.to(device)

            reward, actions_probs, _, user_allocated_props, server_used_props, capacity_used_props, _ \
                = model(user_seq, server_seq, masks, exploration_c)

            if now_train_type == 'REINFORCE':
                if batch_idx == 0:
                    critic_exp_mvg_avg = reward.mean()
                else:
                    critic_exp_mvg_avg = (critic_exp_mvg_avg * train_config['beta']) \
                                         + ((1. - train_config['beta']) * reward.mean())
                advantage = reward - critic_exp_mvg_avg

            elif now_train_type == 'ac':
                critic_reward = critic_model(user_seq, server_seq)
                advantage = reward - critic_reward
                # 训练critic网络
                critic_loss = F.mse_loss(critic_reward, reward.detach()).mean()
                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()

            elif now_train_type == 'RGRB':
                model.policy = 'greedy'
                with torch.no_grad():
                    reward2, _, _, _, _, _, _ = model(user_seq, server_seq, masks, exploration_c)
                    advantage = reward - reward2
                model.policy = 'sample'

            else:
                raise NotImplementedError

            log_probs = torch.zeros(user_seq.size(0), device=device)
            for prob in actions_probs:
                log_prob = torch.log(prob)
                log_probs += log_prob
            log_probs[log_probs < -1000] = -1000.

            reinforce = torch.dot(advantage.detach(), log_probs)
            actor_loss = reinforce.mean()

            optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_config['max_grad_norm']), norm_type=2)
            optimizer.step()

            critic_exp_mvg_avg = critic_exp_mvg_avg.detach()

            if batch_idx % int(1024 / train_config['batch_size']) == 0:
                logger.info(
                    'Epoch {}: Train [{}/{} ({:.1f}%)]\tR:{:.6f}\tuser_props: {:.6f}'
                    '\tserver_props: {:.6f}\tcapacity_props:{:.6f}'.format(
                        epoch,
                        (batch_idx + 1) * len(user_seq),
                        data_config['data_size']['train'],
                        100. * (batch_idx + 1) / len(train_loader),
                        torch.mean(reward),
                        torch.mean(user_allocated_props),
                        torch.mean(server_used_props),
                        torch.mean(capacity_used_props)))

        tensorboard_writer.add_scalar('train/train_reward', torch.mean(reward), epoch)
        tensorboard_writer.add_scalar('train/train_user_allocated_props', torch.mean(user_allocated_props), epoch)
        tensorboard_writer.add_scalar('train/train_server_used_props', torch.mean(server_used_props), epoch)
        tensorboard_writer.add_scalar('train/train_capacity_used_props', torch.mean(capacity_used_props), epoch)

        # Valid and Test
        model.eval()
        model.policy = 'greedy'
        logger.info('')
        with torch.no_grad():
            # Validation
            valid_R_list = []
            valid_user_allocated_props_list = []
            valid_server_used_props_list = []
            valid_capacity_used_props_list = []
            model.policy = 'greedy'
            model.beam_num = 1
            for batch_idx, (server_seq, user_seq, masks) in enumerate(valid_loader):
                server_seq, user_seq, masks = server_seq.to(device), user_seq.to(device), masks.to(device)

                reward, _, _, user_allocated_props, server_used_props, capacity_used_props, _ \
                    = model(user_seq, server_seq, masks, exploration_c)

                if batch_idx % int(1024 / train_config['batch_size']) == 0:
                    logger.info(
                        'Epoch {}: Valid [{}/{} ({:.1f}%)]\tR:{:.6f}\tuser_props: {:.6f}'
                        '\tserver_props: {:.6f}\tcapacity_props:{:.6f}'.format(
                            epoch,
                            (batch_idx + 1) * len(user_seq),
                            data_config['data_size']['valid'],
                            100. * (batch_idx + 1) / len(valid_loader),
                            torch.mean(reward),
                            torch.mean(user_allocated_props),
                            torch.mean(server_used_props),
                            torch.mean(capacity_used_props)
                        ))

                valid_R_list.append(reward)
                valid_user_allocated_props_list.append(user_allocated_props)
                valid_server_used_props_list.append(server_used_props)
                valid_capacity_used_props_list.append(capacity_used_props)

            valid_R_list = torch.cat(valid_R_list)
            valid_user_allocated_props_list = torch.cat(valid_user_allocated_props_list)
            valid_server_used_props_list = torch.cat(valid_server_used_props_list)
            valid_capacity_used_props_list = torch.cat(valid_capacity_used_props_list)
            valid_r = torch.mean(valid_R_list)
            valid_user_allo = torch.mean(valid_user_allocated_props_list)
            valid_server_use = torch.mean(valid_server_used_props_list)
            valid_capacity_use = torch.mean(valid_capacity_used_props_list)
            logger.info('Epoch {}: Valid \tR:{:.6f}\tuser_props: {:.6f}\tserver_props: {:.6f}\tcapacity_props:{:.6f}'
                        .format(epoch, valid_r, valid_user_allo, valid_server_use, valid_capacity_use))

            tensorboard_writer.add_scalar('valid/valid_reward', valid_r, epoch)
            tensorboard_writer.add_scalar('valid/valid_user_allocated_props', valid_user_allo, epoch)
            tensorboard_writer.add_scalar('valid/valid_server_used_props', valid_server_use, epoch)
            tensorboard_writer.add_scalar('valid/valid_capacity_used_props', valid_capacity_use, epoch)

            all_valid_reward_list.append(valid_r)
            all_valid_user_list.append(valid_user_allo)
            all_valid_server_list.append(valid_server_use)
            all_valid_capacity_list.append(valid_capacity_use)

            # 每次遇到更好的reward就保存一次模型
            if valid_r < best_r:
                best_r = valid_r
                best_user = valid_user_allo
                best_server = valid_server_use
                best_capacity = valid_capacity_use
                best_time = 0
                logger.info("目前本次reward最好\n")
                model_filename = dir_name + "/" + time.strftime(
                    '%m%d%H%M', time.localtime(time.time())
                ) + "_{:.2f}_{:.2f}_{:.2f}".format(best_user * 100, best_server * 100, best_capacity * 100) + '.mdl'
                torch.save(model.state_dict(), model_filename)
                logger.info("模型已存储到: {}".format(model_filename))
            else:
                best_time += 1
                logger.info("已经有{}轮效果没变好了\n".format(best_time))

            # Test
            test_R_list = []
            test_user_allocated_props_list = []
            test_server_used_props_list = []
            test_capacity_used_props_list = []
            for batch_idx, (server_seq, user_seq, masks) in enumerate(test_loader):
                server_seq, user_seq, masks = server_seq.to(device), user_seq.to(device), masks.to(device)

                reward, _, _, user_allocated_props, server_used_props, capacity_used_props, _ \
                    = model(user_seq, server_seq, masks, exploration_c)

                if batch_idx % int(1024 / train_config['batch_size']) == 0:
                    logger.info(
                        'Epoch {}: Test [{}/{} ({:.1f}%)]\tR:{:.6f}\tuser_props: {:.6f}'
                        '\tserver_props: {:.6f}\tcapacity_props:{:.6f}'.format(
                            epoch,
                            (batch_idx + 1) * len(user_seq),
                            data_config['data_size']['test'],
                            100. * (batch_idx + 1) / len(valid_loader),
                            torch.mean(reward),
                            torch.mean(user_allocated_props),
                            torch.mean(server_used_props),
                            torch.mean(capacity_used_props)
                        ))

                test_R_list.append(reward)
                test_user_allocated_props_list.append(user_allocated_props)
                test_server_used_props_list.append(server_used_props)
                test_capacity_used_props_list.append(capacity_used_props)

            test_R_list = torch.cat(test_R_list)
            test_user_allocated_props_list = torch.cat(test_user_allocated_props_list)
            test_server_used_props_list = torch.cat(test_server_used_props_list)
            test_capacity_used_props_list = torch.cat(test_capacity_used_props_list)

            test_r = torch.mean(test_R_list)
            test_user_allo = torch.mean(test_user_allocated_props_list)
            test_server_use = torch.mean(test_server_used_props_list)
            test_capacity_use = torch.mean(test_capacity_used_props_list)

            logger.info('Epoch {}: Test \tR:{:.6f}\tuser_props: {:.6f}\tserver_props: {:.6f}\tcapacity_props:{:.6f}'
                        .format(epoch, test_r, test_user_allo, test_server_use, test_capacity_use))
            tensorboard_writer.add_scalar('test/test_reward', test_r, epoch)
            tensorboard_writer.add_scalar('test/test_user_allocated_props', test_user_allo, epoch)
            tensorboard_writer.add_scalar('test/test_server_used_props', test_server_use, epoch)
            tensorboard_writer.add_scalar('test/test_capacity_used_props', test_capacity_use, epoch)

        logger.info('')

        # 如果超过设定的epoch次数valid奖励都没有再提升，就停止训练
        if best_time >= train_config['wait_best_reward_epoch']:
            logger.info("效果如下：")
            for i in range(len(all_valid_reward_list)):
                logger.info("Epoch: {}\treward: {:.6f}\tuser_props: {:.6f}"
                            "\tserver_props: {:.6f}\tcapacity_props: {:.6f}"
                            .format(i, all_valid_reward_list[i], all_valid_user_list[i],
                                    all_valid_server_list[i], all_valid_capacity_list[i]))
            logger.info("训练结束，最好的reward:{}，用户分配率:{:.2f}，服务器租用率:{:.2f}，资源利用率:{:.2f}"
                        .format(best_r, best_user * 100, best_server * 100, best_capacity * 100))

            # 保存一次可继续训练的模型就退出
            now_exit = True

        # 每interval个epoch，或者即将退出的时候，保存一次可继续训练的模型：
        if epoch % train_config['save_model_epoch_interval'] == train_config['save_model_epoch_interval'] - 1 \
                or now_exit:
            model_filename = dir_name + "/" + time.strftime(
                '%m%d%H%M', time.localtime(time.time())
            ) + "_{:.2f}_{:.2f}_{:.2f}".format(valid_user_allo * 100,
                                               valid_server_use * 100,
                                               valid_capacity_use * 100) + '.pt'
            if now_train_type == 'ac':
                state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch,
                         'critic_model': critic_model.state_dict(),
                         'critic_optimizer': critic_optimizer.state_dict()}
            else:
                state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
            torch.save(state, model_filename)
            logger.info("模型已存储到: {}".format(model_filename))

            if now_exit:
                if original_train_type == 'REINFORCE+RGRB':
                    if now_train_type == 'REINFORCE':
                        now_train_type = 'RGRB'
                        logger.info("REINFORCE无进步，已切换训练方式为RGRB")
                        now_exit = False
                        best_time = 0
                    else:
                        return model_filename
                else:
                    return model_filename


if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        loaded_config = yaml.safe_load(f)
    train(loaded_config)

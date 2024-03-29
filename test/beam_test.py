from torch.utils.data import DataLoader
import torch
from nets.attention_net import AttentionNet
import pickle
import time

batch_size = 32
beam_num = 5
use_cuda = True
lr = 1e-4
beta = 0.9
max_grad_norm = 2.
epochs = 1000
capacity_reward_rate = 0.2
user_num = 200
resource_rate = 3
x_end = 0.5
y_end = 1
min_cov = 1
max_cov = 1.5
miu = 35
sigma = 10
user_embedding_type = 'transformer'
server_embedding_type = 'linear'
test_size = 10000
wait_best_reward_epoch = 20
save_model_epoch_interval = 10
use_cuda = use_cuda and torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

test_filename = "D:/transformer_eua/dataset/test_server_" + str(x_end) + "_" + str(y_end) + "_user_" \
                    + str(user_num) + "_miu_" + str(miu) + "_sigma_" + str(sigma) + "_size_" + str(test_size) + ".pkl"
print("正在加载测试数据集")
with open(test_filename, 'rb') as f:
    test_set = pickle.load(f)

print("正在加载模型")
model = AttentionNet(6, 7, 256, device=device, capacity_reward_rate=capacity_reward_rate,
                     user_embedding_type=user_embedding_type, server_embedding_type=server_embedding_type)

model_filename = "D:/transformer_eua/model/" \
                 "03111924_server_0.5_1_user_200_miu_35_sigma_10_transformer_linear_RGRB/03122134_96.32_55.01_51.61.mdl"
checkpoint = torch.load(model_filename)
model.load_state_dict(checkpoint['model'])

test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)

print("开始测试")
# 测试beam_search的效果
beam_R_list = []
beam_user_allocated_props_list = []
beam_server_used_props_list = []
beam_capacity_used_props_list = []
model.policy = 'greedy'
model.beam_num = beam_num
with torch.no_grad():
    for batch_idx, (server_seq, user_seq, masks) in enumerate(test_loader):
        server_seq, user_seq, masks = server_seq.to(device), user_seq.to(device), masks.to(device)

        reward, _, action_idx, user_allocated_props, server_used_props, capacity_used_props, user_allocate_list \
            = model(user_seq, server_seq, masks)

        beam_R_list.append(reward)
        beam_user_allocated_props_list.append(user_allocated_props)
        beam_server_used_props_list.append(server_used_props)
        beam_capacity_used_props_list.append(capacity_used_props)

        print('{} Test [{}/{} ({:.1f}%)]\tR:{:.6f}\tuser_props: {:.6f}\tserver_props: {:.6f}'
              '\tcapacity_props:{:.6f}'.format(time.strftime('%H:%M:%S', time.localtime(time.time())),
                                               (batch_idx + 1) * len(user_seq),
                                               test_size,
                                               100. * (batch_idx + 1) / len(test_loader),
                                               torch.mean(reward),
                                               torch.mean(user_allocated_props),
                                               torch.mean(server_used_props),
                                               torch.mean(capacity_used_props)))

    beam_R_list = torch.cat(beam_R_list)
    beam_user_allocated_props_list = torch.cat(beam_user_allocated_props_list)
    beam_server_used_props_list = torch.cat(beam_server_used_props_list)
    beam_capacity_used_props_list = torch.cat(beam_capacity_used_props_list)
    r = torch.mean(beam_R_list)
    beam_user_allo = torch.mean(beam_user_allocated_props_list)
    beam_server_use = torch.mean(beam_server_used_props_list)
    beam_capacity_use = torch.mean(beam_capacity_used_props_list)

    print('{} BeamTest \tR:{:.6f}\tuser_props: {:.6f}'
          '\tserver_props: {:.6f}\tcapacity_props:{:.6f}'
          .format(time.strftime('%H:%M:%S', time.localtime(time.time())), r, beam_user_allo,
                  beam_server_use, beam_capacity_use))

import sys
import numpy as np
import random
import math
from model import W_Net
from eval.eval_by_cropping import eval_model
from Dataset.TrainDatasetConstructor import TrainDatasetConstructor
from Dataset.EvalDatasetConstructor import EvalDatasetConstructor
from metrics import JointLoss, AEBatch, SEBatch
from PIL import Image
import time
import torch
# torch.backends.cudnn.benchmark=True
# config
config = {
'SHANGHAITECH': 'B',
'min_RATE':10000000,
'min_MAE':10240000,
'min_MSE':10240000,
'eval_num':316,
'train_num':400,
'learning_rate': 1e-6,
'train_batch_size': 10,
'epoch': 10000,
'eval_per_step': 200,
'mode':'crop'
}
img_dir = "/home/zzn/Documents/Datasets/part_" + config['SHANGHAITECH'] + "_final/train_data/images"
gt_dir = "/home/zzn/Documents/Datasets/part_" + config['SHANGHAITECH'] + "_final/train_data/gt_map_w_net"
binary_dir = "/home/zzn/Documents/Datasets/part_" + config['SHANGHAITECH'] + "_final/train_data/blur_map_w_net"
img_dir_t = "/home/zzn/Documents/Datasets/part_" + config['SHANGHAITECH'] + "_final/test_data/images"
gt_dir_t = "/home/zzn/Documents/Datasets/part_" + config['SHANGHAITECH'] + "_final/test_data/gt_map_w_net"
binary_dir_t = "/home/zzn/Documents/Datasets/part_" + config['SHANGHAITECH'] + "_final/test_data/blur_map_w_net"
model_save_path = "/home/zzn/PycharmProjects/W-Net_pytorch/checkpoints/model_w_net_B.pkl"
f = open("/home/zzn/PycharmProjects/W-Net_pytorch/logs/log_w_net_B.txt", "w")
# data_load
train_dataset = TrainDatasetConstructor(img_dir, gt_dir, binary_dir, config['train_num'], mode=config['mode'], if_random_hsi=True, if_flip=True)
eval_dataset = EvalDatasetConstructor(img_dir_t, gt_dir_t, config['eval_num'], mode=config['mode'])
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=config['train_batch_size'])
eval_loader = torch.utils.data.DataLoader(dataset=eval_dataset, batch_size=1)

# obtain the gpu device
assert torch.cuda.is_available()
cuda_device = torch.device("cuda")

# model construct
net = W_Net().cuda()
encoder = list(net.children())[0]
decoder = list(net.children())[1]
attention = list(net.children())[2]
output = list(net.children())[3]
# net = torch.load("/home/zzn/PycharmProjects/W-Net_pytorch/checkpoints/model_w_net.pkl")
# set optimizer and estimator

optimizer_1 = torch.optim.Adam([{'params': encoder.parameters()}, {'params': decoder.parameters()}, {'params': output.parameters()}], config['learning_rate'], weight_decay=5e-3)
optimizer_2 = torch.optim.Adam(attention.parameters(), config['learning_rate'], weight_decay=5e-3)
# criterion = JointLoss(alpha=100000, beta=6).cuda()
criterion_mseloss = torch.nn.MSELoss(size_average=True).cuda()
criterion_bceloss = torch.nn.BCELoss(size_average=True).cuda()
ae_batch = AEBatch().cuda()
se_batch = SEBatch().cuda()
modules = {'model':net, 'ae':ae_batch, 'se':se_batch}

step = 0
# torch.cuda.empty_cache()
for epoch_index in range(config['epoch']):
    dataset = train_dataset.shuffle()
    mse_loss_list = []
    bce_loss_list = []
    time_per_epoch = 0
    
    for train_img_index, train_img, train_gt, train_binary in train_loader:
        if step % config['eval_per_step'] == 0:
            validate_MAE, validate_RMSE, time_cost = eval_model(config, eval_loader, modules, False)
            
            f.write('In step {}, epoch {},  MAE = {}, MSE = {}, time cost eval = {}s\n'.format(step, epoch_index + 1, validate_MAE, validate_RMSE, time_cost))
            f.flush()
            
#             save model
            if config['min_MAE'] > validate_MAE:
                config['min_MAE'] = validate_MAE
                torch.save(net, model_save_path)
#             torch.save(net, "/home/zzn/Downloads/CSRNet_pytorch-master/checkpoints/model_in_time.pkl")
            
            # return train model
        net.train()
        torch.cuda.empty_cache()
        
#         loss = criterion(prediction, y, z)
        optimizer_1.zero_grad()
        optimizer_2.zero_grad()
                # B
        x = train_img
        y = train_gt
        z = train_binary
        start = time.time()
        
        B5_C3, B4_C3, B3_C3, B2_C2 = encoder(x)
        decoder_map = decoder(B5_C3, B4_C3, B3_C3, B2_C2)
        attention_map = attention(B5_C3, B4_C3, B3_C3, B2_C2)
        prediction = output(torch.mul(decoder_map, attention_map))
        
        bce_loss = criterion_bceloss(attention_map, z) * 20
        mse_loss = criterion_mseloss(prediction, y) * 1000
        mse_loss_list.append(mse_loss.data.item())
        bce_loss_list.append(bce_loss.data.item())
        bce_loss.backward(retain_graph=True)
        mse_loss.backward() # calculate gradient
        optimizer_1.step() # update encoder-decorder architecture's parameters
        optimizer_2.step() # update reinforcement architecture's parameters
        step += 1
        torch.cuda.synchronize()
        end2 = time.time()
        time_per_epoch += end2 - start

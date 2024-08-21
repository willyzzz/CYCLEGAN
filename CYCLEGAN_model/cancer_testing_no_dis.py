import torch
import os
import numpy as np
from dataset_input import cancer_dataset_process
from utils import ccc
from no_dis_model_structure import *
import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

# 设置随机种子和设备
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fra_ini_icon = 'sampling'

sig_path = '../cancer_data/sig.txt'
test_bulk_path = '../cancer_data/2015_bulk_rna_seq.txt'
test_fra_gt_path = '../cancer_data/2015_fraction_gt.txt'
ini_fra_path_folder  = Path('../cancer_data/samples_fraction_ini')
ini_fra_paths = [str(file) for file in ini_fra_path_folder.iterdir() if file.is_file()]

model_save_path = './model_checkpoints'
testing_dataset_name = '2015cell_TCGA'

data_path_list = [sig_path, test_bulk_path, test_fra_gt_path, ini_fra_paths]
dataloader, bulk, sig, fra_gt, fra_ini = cancer_dataset_process(data_path_list, log_icon=True, add_noise=False)

# 获取数据维度
cell_type_dim = fra_ini.shape[1]
gene_dim = sig.shape[1]

# 初始化模型
G_sc_bulk = Generator_sc_to_bulk(cell_type_dim, gene_dim, device).to(device)
G_bulk_sc = Generator_bulk_to_sc(cell_type_dim, gene_dim, device).to(device)
D_bulk = Discriminator_bulk(gene_dim, device).to(device)

# 定义损失函数和优化器
learning_rate = 1e-4
num_epochs = 10000
criterion_GAN = nn.BCEWithLogitsLoss()
criterion_cycle = nn.L1Loss()
lambda_cycle = 1
lambda_full_fra = 5
lamda_GAN = 1

optimizer_G_sc_bulk = optim.RMSprop(G_sc_bulk.parameters(), lr=learning_rate)
optimizer_G_bulk_sc = optim.RMSprop(G_bulk_sc.parameters(), lr=learning_rate)
optimizer_D_bulk = optim.RMSprop(D_bulk.parameters(), lr=learning_rate)

# 切换到训练模式
G_sc_bulk.train()
G_bulk_sc.train()
D_bulk.train()

G_losses = []
D_losses = []

# 初始化列表来保存每个epoch的ccc值
ccc_values = []
best_ccc = -float('inf')
best_epoch = -1
best_fake_G1_full_fra_cat = None
best_fake_bulk_cat = None

# 训练循环中的每个epoch保存ccc值
for epoch in range(num_epochs):
    total_G_loss = 0.0
    total_D_loss = 0.0

    fake_G1_full_fra_cat = torch.empty(0, cell_type_dim).to(device)
    recon_G1_full_fra_cat = torch.empty(0, cell_type_dim).to(device)

    fake_bulk_cat = torch.empty(0, gene_dim).to(device)

    for i, data in enumerate(dataloader):
        real_bulk_data = data[0].to(torch.float).to(device)
        real_sig_data = sig.to(torch.float).to(device)
        real_fra_data = data[1].to(torch.float).to(device)

        fake_bulk = G_sc_bulk(real_sig_data, real_fra_data)
        fake_fra = G_bulk_sc(real_bulk_data)

        # Cycle consistency losses
        recon_bulk = G_sc_bulk(real_sig_data, fake_fra)
        recon_fra = G_bulk_sc(fake_bulk)

        set_requires_grad([D_bulk], False)

        optimizer_G_sc_bulk.zero_grad()
        optimizer_G_bulk_sc.zero_grad()

        # 计算对抗损失
        pred_fake_bulk = D_bulk(fake_bulk)

        loss_GAN_bulk = criterion_GAN(pred_fake_bulk, torch.ones_like(pred_fake_bulk))

        # 计算循环一致性损失
        loss_cycle_bulk = criterion_cycle(recon_bulk, real_bulk_data)
        loss_cycle_fra = criterion_cycle(recon_fra, real_fra_data)

        # 计算最终的生成器损失
        loss_G = lamda_GAN * loss_GAN_bulk + lambda_cycle * (loss_cycle_bulk+ loss_cycle_fra)
        loss_G.backward()
        optimizer_G_sc_bulk.step()
        optimizer_G_bulk_sc.step()

        # 计算判别器损失
        set_requires_grad([D_bulk], True)
        optimizer_D_bulk.zero_grad()

        pred_real_bulk = D_bulk(real_bulk_data)
        loss_D_real_bulk = criterion_GAN(pred_real_bulk, torch.ones_like(pred_real_bulk))

        pred_fake_bulk = D_bulk(fake_bulk.detach())
        loss_D_fake_bulk = criterion_GAN(pred_fake_bulk, torch.zeros_like(pred_fake_bulk))

        loss_D_bulk = loss_D_real_bulk + loss_D_fake_bulk

        # 总判别器损失
        loss_D = loss_D_bulk
        loss_D.backward()
        optimizer_D_bulk.step()

        total_G_loss += loss_G.item()
        total_D_loss += loss_D.item()

        fake_G1_full_fra_cat = torch.cat((fake_G1_full_fra_cat, fake_fra), dim=0)
        recon_G1_full_fra_cat = torch.cat((recon_G1_full_fra_cat, recon_fra), dim=0)
        fake_bulk_cat = torch.cat((fake_bulk_cat, fake_bulk), dim=0)

    # 打印每个 epoch 的平均损失
    avg_G_loss = total_G_loss / len(dataloader)
    avg_D_loss = total_D_loss / len(dataloader)

    G_losses.append(avg_G_loss)
    D_losses.append(avg_D_loss)

    # 计算当前epoch的ccc值
    current_ccc = ccc(fake_G1_full_fra_cat, fra_gt)
    ccc_values.append(current_ccc)

    if np.isnan(current_ccc):
        print(f"NaN CCC value encountered at epoch {epoch}. Terminating training.")
        break

    # 保存最好ccc值和对应的生成结果
    if current_ccc > best_ccc:
        best_ccc = current_ccc
        best_epoch = epoch
        best_fake_G1_full_fra_cat = fake_G1_full_fra_cat.clone().detach()
        best_fake_bulk_cat = fake_bulk_cat.clone().detach()

    if epoch % 100 == 0:
        print('[Epoch %d/%d] Average G loss: %.4f, Average D loss: %.4f' % (
            epoch + 1, num_epochs, avg_G_loss, avg_D_loss))
        print(f"loss_GAN_bulk: {loss_GAN_bulk}")
        print(f"loss_cycle_bulk: {loss_cycle_bulk}")
        print(f"loss_cycle_fra: {loss_cycle_fra}")
        print(f"loss_D_bulk: {loss_D_bulk}")
        print('*********performance_showing**********')
        print('fra_G1_ccc = ', current_ccc)
        print('*********gan_performance_showing**********')
        print('bulk_ccc = ', ccc(fake_bulk_cat, bulk))
        print('fake_G1_full_fra_cat:', fake_G1_full_fra_cat)
        print('\n')
        print('fake bulk:', fake_bulk_cat)
        print('fake fra', fake_G1_full_fra_cat)

# 画出ccc曲线
plt.plot(range(len(ccc_values)), ccc_values)
plt.xlabel('Epoch')
plt.ylabel('CCC')
plt.title('CCC Curve')
plt.show()

plt.figure(figsize=(10, 5))
# Plot Generator Loss
plt.plot(range(len(G_losses)), G_losses, label="Generator Loss")
# Plot Discriminator Loss
plt.plot(range(len(D_losses)), D_losses, label="Discriminator Loss")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Generator and Discriminator Losses During Training')
plt.legend()
plt.show()

# 找出最好的ccc值
print('Best CCC:', best_ccc)
print('Best Epoch:', best_epoch)
print('Best fake_G1_full_fra_cat:', best_fake_G1_full_fra_cat)
print('Best fake_bulk_cat:', best_fake_bulk_cat)
pd.DataFrame(best_fake_G1_full_fra_cat.cpu().numpy()).to_csv('./best_fra.txt',sep='\t')

# 测试数据运行
G_sc_bulk.eval()
G_bulk_sc.eval()
D_bulk.eval()

with torch.no_grad():
    fake_bulk_test = G_sc_bulk(sig.to(torch.float).to(device), fra_ini.to(torch.float).to(device))
    fake_fra_test = G_bulk_sc(bulk.to(torch.float).to(device))
    print('Test fake_bulk:', fake_bulk_test)
    print('Test fake_fra:', fake_fra_test)
    print('Test bulk ccc: ', ccc(fake_bulk_test, bulk))
    print('Test fra ccc: ', ccc(fake_fra_test, fra_gt))
    # 在这里可以添加其他评估指标的计算和打印

import torch
import torch.nn as nn
import torch.nn.functional as F


class normalized_1_sum(nn.Module):
    def __init__(self, dim):
        super(normalized_1_sum, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x / x.sum(dim=self.dim, keepdim=True)

class Generator_sc_to_bulk(nn.Module):
    def __init__(self, cell_type_dim, gene_dim, device):
        super(Generator_sc_to_bulk, self).__init__()
        self.device = device
        self.bulk_denoise = nn.Sequential(
            nn.Linear(gene_dim, 2048),
            nn.ELU(),
            nn.Dropout(),

            nn.Linear(2048, 1024),
            nn.ELU(),
            nn.Dropout(),

            nn.Linear(1024, 256),
            nn.ELU(),
            nn.Dropout(),

            nn.Linear(256, 1024),
            nn.ELU(),
            nn.Dropout(),

            nn.Linear(1024, 2048),
            nn.ELU(),
            nn.Dropout(),

            nn.Linear(2048, gene_dim),
            nn.Softplus()
        )
        self.to(device)


    def forward(self, sig, fra):
        fake_bulk = torch.matmul(fra, sig)
        fake_bulk = self.bulk_denoise(fake_bulk)
        return fake_bulk


import torch
import torch.nn as nn

def normalized_1_sum(x, dim):
    return x / x.sum(dim=dim, keepdim=True)

class SelfAttention(nn.Module):
    def __init__(self, gene_dim, heads=128):
        super(SelfAttention, self).__init__()
        self.heads = heads
        self.original_gene_dim = gene_dim

        # 计算新的 gene_dim，使其可以被 heads 整除
        self.padded_gene_dim = (gene_dim + heads - 1) // heads * heads

        self.query = nn.Linear(self.padded_gene_dim, self.padded_gene_dim)
        self.key = nn.Linear(self.padded_gene_dim, self.padded_gene_dim)
        self.value = nn.Linear(self.padded_gene_dim, self.padded_gene_dim)
        self.scale = (self.padded_gene_dim // heads) ** -0.5

    def forward(self, x):
        batch_size = x.size(0)

        # 对 gene_dim 进行填充
        x = F.pad(x, (0, self.padded_gene_dim - self.original_gene_dim))

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        Q = Q.view(batch_size, self.heads, -1)
        K = K.view(batch_size, self.heads, -1)
        V = V.view(batch_size, self.heads, -1)

        attention = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attention = torch.softmax(attention, dim=-1)

        x = torch.matmul(attention, V).view(batch_size, -1)

        # 移除填充部分
        x = x[:, :self.original_gene_dim]
        return x


class Generator_bulk_to_sc(nn.Module):
    def __init__(self, cell_type_dim, gene_dim, device, lambda_reg=0.1):
        super(Generator_bulk_to_sc, self).__init__()

        self.attention = SelfAttention(gene_dim)

        self.compress = nn.Sequential(
            nn.Linear(gene_dim, 512),  # 压缩到中间维度
            nn.ELU(),
            nn.Dropout(),

            nn.Linear(512, cell_type_dim),  # 压缩到 cell_type_dim
            nn.Softplus()
        )

        self.lambda_reg = lambda_reg
        self.device = device
        self.to(device)

    def forward(self, bulk):
        attended_bulk = self.attention(bulk)  # 通过注意力机制提取特征
        fake_fra = self.compress(attended_bulk)  # 维度压缩到 cell_type_dim
        fake_fra = normalized_1_sum(fake_fra, dim=1)  # 归一化输出
        return fake_fra



class Discriminator_bulk(nn.Module):
    def __init__(self, gene_dim, device):
        super(Discriminator_bulk, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(gene_dim,  1024),
            nn.LeakyReLU(),

            nn.Linear(1024, 512),
            nn.LeakyReLU(),

            nn.Linear(512, 256),
            nn.LeakyReLU(),

            nn.Linear(256, 1),
        )
        self.device = device
        self.to(device)

    def forward(self, x):
        return self.model(x)


class Discriminator_sc(nn.Module):
    def __init__(self, cell_type_dim, gene_dim, device):
        super(Discriminator_sc, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(cell_type_dim, 1)
        )
        self.device = device
        self.to(device)

    def forward(self, x):
        return self.model(x)


def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad
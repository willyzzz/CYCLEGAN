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
            nn.Linear(2048, 1024),
            nn.ELU(),
            nn.Linear(1024, 256),
            nn.ELU(),
            nn.Linear(256, 1024),
            nn.ELU(),
            nn.Linear(1024, 2048),
            nn.ELU(),
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

class Generator_bulk_to_sc(nn.Module):
    def __init__(self, cell_type_dim, gene_dim, device, lambda_reg=0.1):
        super(Generator_bulk_to_sc, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(gene_dim, 1024),
            nn.ELU(),
            nn.Linear(1024, 256),
            nn.ELU(),
            nn.Linear(256, 32),
            nn.ELU(),
            nn.Linear(32, cell_type_dim),
            nn.Softplus()
        )
        self.lambda_reg = lambda_reg
        self.device = device
        self.to(device)

    def forward(self, bulk):
        fake_fra = self.model(bulk)
        fake_fra = normalized_1_sum(fake_fra, dim=1)

        # # 使用QR分解求解
        # Q, R = torch.linalg.qr(fake_fra)
        #
        # try:
        #     # 尝试计算解
        #     fake_sig = torch.linalg.solve(R, torch.matmul(Q.T, bulk))
        # except torch.linalg.LinAlgError as e:
        #
        #     # # 方法1: 使用伪逆
        #     R_pseudo_inv = torch.linalg.pinv(R)
        #     fake_sig = torch.matmul(R_pseudo_inv, torch.matmul(Q.T, bulk))
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
            nn.Linear(256, 1)
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
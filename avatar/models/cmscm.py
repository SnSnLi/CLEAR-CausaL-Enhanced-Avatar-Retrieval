import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple

class CausalVariable(nn.Module):
    """表示SCM中的变量"""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.noise_encoder = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        
    def add_noise(self, x: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(x)
        return x + self.noise_encoder(noise)

class CMSCM(nn.Module):
    def __init__(self, hidden_dim=768, shared_dim=256):
        super().__init__()
        # 变量定义
        self.X = CausalVariable(hidden_dim)
        self.Y = CausalVariable(hidden_dim)
        self.S = CausalVariable(shared_dim)
        
        # 结构方程
        self.fx = nn.Sequential(
            nn.Linear(shared_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        self.fy = nn.Sequential(
            nn.Linear(shared_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        self.fs = nn.Sequential(
            nn.Linear(hidden_dim * 2, shared_dim),
            nn.LayerNorm(shared_dim),
            nn.ReLU()
        )
        self.fzx = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        self.fzy = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        self.fr = nn.Sequential(
            nn.Linear(shared_dim, 1),
            nn.Sigmoid()
        )

    def structural_equations(self, X, Y):
        """计算结构方程"""
        # 添加噪声
        X_noised = self.X.add_noise(X)
        Y_noised = self.Y.add_noise(Y)
        
        # 计算共享语义
        S = self.fs(torch.cat([X_noised, Y_noised], dim=-1))
        
        # 计算模态特定语义
        Zx = self.fzx(X_noised)
        Zy = self.fzy(Y_noised)
        
        # 计算重建值
        X_hat = self.fx(torch.cat([S, Zx], dim=-1))
        Y_hat = self.fy(torch.cat([S, Zy], dim=-1))
        
        # 计算相关性得分
        R = self.fr(S)
        
        return {
            'S': S,
            'Zx': Zx,
            'Zy': Zy,
            'X_hat': X_hat,
            'Y_hat': Y_hat,
            'R': R
        }

    def frontdoor_adjustment(self, X, Y):
        """前门准则计算"""
        with torch.no_grad():
            S_given_X = self.fs(torch.cat([X, torch.zeros_like(Y)], dim=-1))
            Y_given_S = self.fy(torch.cat([S_given_X, self.fzy(Y)], dim=-1))
            effect = Y_given_S.mean(dim=0)
        return effect

    def backdoor_adjustment(self, X, Y):
        """后门调整计算"""
        with torch.no_grad():
            S = self.fs(torch.cat([X, Y], dim=-1))
            Zx = self.fzx(X)
            Zy = self.fzy(Y)
            joint = torch.cat([S, Zx, Zy], dim=-1)
            Y_given_XSZ = self.fy(torch.cat([S, Zy], dim=-1))
            effect = (Y_given_XSZ * joint).mean(dim=0)
        return effect

    def compute_effects(self, X, Y):
        """计算直接效应和间接效应"""
        # 直接效应(前门)
        direct_effect = self.frontdoor_adjustment(X, Y)
        
        # 间接效应(后门-前门)
        total_effect = self.backdoor_adjustment(X, Y)
        indirect_effect = total_effect - direct_effect
        
        return direct_effect, indirect_effect

    def compute_losses(self, X, Y, outputs):
        """计算损失函数"""
        # 重建损失
        recon_loss = F.mse_loss(outputs['X_hat'], X) + F.mse_loss(outputs['Y_hat'], Y)
        
        # 因果一致性损失
        direct_effect, indirect_effect = self.compute_effects(X, Y)
        consistency_loss = F.mse_loss(
            direct_effect + indirect_effect,
            outputs['R'].squeeze()
        )
        
        # 不变性损失
        invariance_loss = F.mse_loss(outputs['Zx'], outputs['Zy'])
        
        return {
            'recon_loss': recon_loss,
            'consistency_loss': consistency_loss,
            'invariance_loss': invariance_loss
        }

    def forward(self, X, Y):
        """前向传播"""
        # 计算结构方程
        outputs = self.structural_equations(X, Y)
        
        # 计算因果效应
        direct_effect, indirect_effect = self.compute_effects(X, Y)
        
        # 计算损失
        losses = self.compute_losses(X, Y, outputs)
        
        return {
            **outputs,
            'direct_effect': direct_effect,
            'indirect_effect': indirect_effect,
            **losses
        }

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple

class CausalVariable(nn.Module):
    """Represents a variable in a Structural Causal Model (SCM)."""
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
        """Add noise to the input tensor using a learned noise encoder."""
        noise = torch.randn_like(x)
        return x + self.noise_encoder(noise)

class CMSCM(nn.Module):
    def __init__(self, hidden_dim=768, shared_dim=256):
        super().__init__()
        # Variable definitions
        self.X = CausalVariable(hidden_dim)
        self.Y = CausalVariable(hidden_dim)
        self.S = CausalVariable(shared_dim)
        
        # Structural equations
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
        """Compute the structural equations for the given inputs."""
        # Add noise to inputs
        X_noised = self.X.add_noise(X)
        Y_noised = self.Y.add_noise(Y)
        
        # Compute shared semantics
        S = self.fs(torch.cat([X_noised, Y_noised], dim=-1))
        
        # Compute modality-specific semantics
        Zx = self.fzx(X_noised)
        Zy = self.fzy(Y_noised)
        
        # Compute reconstructed values
        X_hat = self.fx(torch.cat([S, Zx], dim=-1))
        Y_hat = self.fy(torch.cat([S, Zy], dim=-1))
        
        # Compute relevance score
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
        """Compute the front-door adjustment for causal effect estimation."""
        with torch.no_grad():
            S_given_X = self.fs(torch.cat([X, torch.zeros_like(Y)], dim=-1))
            Y_given_S = self.fy(torch.cat([S_given_X, self.fzy(Y)], dim=-1))
            effect = Y_given_S.mean(dim=0)
        return effect

    def backdoor_adjustment(self, X, Y):
        """Compute the back-door adjustment for causal effect estimation."""
        with torch.no_grad():
            S = self.fs(torch.cat([X, Y], dim=-1))
            Zx = self.fzx(X)
            Zy = self.fzy(Y)
            joint = torch.cat([S, Zx, Zy], dim=-1)
            Y_given_XSZ = self.fy(torch.cat([S, Zy, X], dim=-1))
            effect = (Y_given_XSZ * joint).mean(dim=0)
        return effect

    def compute_effects(self, X, Y):
        """Compute direct and indirect causal effects."""
        # Direct effect (front-door)
        direct_effect = self.frontdoor_adjustment(X, Y)
        
        # Indirect effect (back-door - front-door)
        total_effect = self.backdoor_adjustment(X, Y)
        indirect_effect = total_effect - direct_effect
        
        return direct_effect, indirect_effect

    def compute_losses(self, X, Y, outputs):
        """Compute the loss functions for training."""
        # Reconstruction loss
        recon_loss = F.mse_loss(outputs['X_hat'], X) + F.mse_loss(outputs['Y_hat'], Y)
        
        # Causal consistency loss
        direct_effect, indirect_effect = self.compute_effects(X, Y)
        consistency_loss = F.mse_loss(
            direct_effect + indirect_effect,
            outputs['R'].squeeze()
        )
        
        # Invariance loss
        invariance_loss = F.mse_loss(outputs['Zx'], outputs['Zy'])
        
        return {
            'recon_loss': recon_loss,
            'consistency_loss': consistency_loss,
            'invariance_loss': invariance_loss
        }

    def forward(self, X, Y):
        """Forward pass through the model."""
        # Compute structural equations
        outputs = self.structural_equations(X, Y)
        
        # Compute causal effects
        direct_effect, indirect_effect = self.compute_effects(X, Y)
        
        # Compute losses
        losses = self.compute_losses(X, Y, outputs)
        
        return {
            **outputs,
            'direct_effect': direct_effect,
            'indirect_effect': indirect_effect,
            **losses
        }

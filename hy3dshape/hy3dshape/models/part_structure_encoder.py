import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import T5Tokenizer, T5EncoderModel  # Text cond
from transformers import AutoImageProcessor, AutoModel  # DINOv2 for image

# Part definition for prosthetic cover
PARTS = {
    'external_top':    0,  # Upper aesthetic shell
    'external_bottom': 1,  # Lower aesthetic shell  
    'internal_cavity': 2,  # MUST fit stump measurements
    'connector':       3,  # Attachment points
}

class PartStructureEncoder(nn.Module):
    """
    Encodes the 3D prior's part structure into
    per-part conditioning tokens for the DiT.
    """
    def __init__(self, cond_dim=1536, n_parts=4):
        super().__init__()
        # Each part gets its own measurement constraints
        self.part_measurement_proj = nn.ModuleList([
            nn.Linear(6, cond_dim) for _ in range(n_parts)
            # 6 = [extent_x, extent_y, extent_z, 
            #      centroid_x, centroid_y, centroid_z]
        ])
        # Part importance weights (cavity > exterior for fit)
        self.part_importance = nn.Parameter(
            torch.tensor([0.3, 0.3, 1.0, 0.4])  # cavity weighted highest
        )
    
    def forward(self, part_measurements: dict):
        """
        part_measurements: {
            'external_top':    [B, 6],
            'external_bottom': [B, 6],
            'internal_cavity': [B, 6],  # ← most critical
            'connector':       [B, 6],
        }
        Returns: [B, n_parts, cond_dim] — one token per part
        """
        tokens = []
        for i, part_name in enumerate(PARTS.keys()):
            measurements = part_measurements[part_name]
            token = self.part_measurement_proj[i](measurements)
            token = token * self.part_importance[i]  # weight by importance
            tokens.append(token.unsqueeze(1))
        return torch.cat(tokens, dim=1)  # [B, 4, cond_dim]


    
class PartStructureEncoderMeshStyle(nn.Module):
    def __init__(self, cond_dim=1536, n_parts=4):
        super().__init__()
        self.part_measurement_proj = nn.ModuleList([
            nn.Linear(6, cond_dim) for _ in range(n_parts)
        ])
        
        # NEW: stress/importance is INPUT, not a fixed parameter
        # MechStyle insight: importance should come from simulation,
        # not be hand-tuned
        self.stress_proj = nn.Linear(1, cond_dim)  # scalar stress → token modulator
        
        # Learned blend between measurement and stress signal
        self.stress_gate = nn.Sequential(
            nn.Linear(cond_dim * 2, cond_dim),
            nn.Sigmoid()
        )

    def forward(self, part_measurements: dict, part_stress: dict = None):
        """
        part_stress: optional dict of per-part scalar stress values
            e.g. {
                'internal_cavity': 0.95,  # high — must not deform
                'connector':       0.80,  # high — load bearing
                'external_top':    0.15,  # low  — free to stylize
                'external_bottom': 0.20,  # low  — free to stylize
            }
        If no FEA available: fall back to fixed heuristic weights.
        """
        tokens = []
        for i, part_name in enumerate(PARTS.keys()):
            meas_token = self.part_measurement_proj[i](
                part_measurements[part_name]  # [B, 6]
            )  # [B, cond_dim]
            
            if part_stress is not None:
                # MechStyle-style: stress gates how strongly
                # measurements constrain this part
                stress_val = torch.tensor(
                    [[part_stress[part_name]]],
                    device=meas_token.device,
                    dtype=meas_token.dtype
                ).expand(meas_token.shape[0], 1)  # [B, 1]
                
                stress_emb = self.stress_proj(stress_val)  # [B, cond_dim]
                
                # High stress → token emphasizes measurement constraint
                # Low stress  → token emphasizes style freedom
                gate = self.stress_gate(
                    torch.cat([meas_token, stress_emb], dim=-1)
                )  # [B, cond_dim], values 0→1
                
                meas_token = gate * meas_token  # stress-weighted
            
            tokens.append(meas_token.unsqueeze(1))
        
        return torch.cat(tokens, dim=1)  # [B, 4, cond_dim]
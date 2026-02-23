#!/usr/bin/env python3
"""
QGT with Quantum Autoencoder Compression (10 → 6 qubits)
Fixed version - handles NaN in amplitude normalization.

Usage:
    python train_qae.py --data_path data.pkl --qae_epochs 50 --epochs 50
"""

import os
import json
import pickle
import argparse
import math
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR
from tqdm import tqdm

import pennylane as qml

from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax as pyg_softmax
from torch_scatter import scatter_add


# =============================================================================
# Quantum Device Setup
# =============================================================================
N_QUBITS_INPUT = 10
N_QUBITS_LATENT = 6
N_QUBITS_TRASH = N_QUBITS_INPUT - N_QUBITS_LATENT

print("Setting up quantum devices...")

try:
    dev_input = qml.device("lightning.qubit", wires=N_QUBITS_INPUT)
    dev_latent = qml.device("lightning.qubit", wires=N_QUBITS_LATENT)
    QUANTUM_DEVICE = "lightning.qubit"
    DIFF_METHOD = "adjoint"
except Exception as e:
    dev_input = qml.device("default.qubit", wires=N_QUBITS_INPUT)
    dev_latent = qml.device("default.qubit", wires=N_QUBITS_LATENT)
    QUANTUM_DEVICE = "default.qubit"
    DIFF_METHOD = "backprop"

print(f"QAE device: {QUANTUM_DEVICE} ({DIFF_METHOD})")
print(f"QGT device: {QUANTUM_DEVICE} ({DIFF_METHOD})")


# =============================================================================
# Config
# =============================================================================
class Config:
    def __init__(self):
        self.seed = 42
        self.emb_dim = 768
        self.n_qubits_input = N_QUBITS_INPUT
        self.n_qubits_latent = N_QUBITS_LATENT
        self.amp_dim_input = 2 ** N_QUBITS_INPUT
        self.amp_dim_latent = 2 ** N_QUBITS_LATENT
        self.n_gnn_layers = 1

        # QAE Training
        self.qae_epochs = 50
        self.qae_lr = 0.001
        self.qae_batch_size = 32
        self.qae_n_layers = 2
        self.qae_scheduler = "plateau"
        self.qae_patience = 5
        self.qae_min_lr = 1e-5
        self.qae_warmup_epochs = 3
        self.qae_max_samples = None

        # QGT Training
        self.batch_size = 8
        self.epochs = 50
        self.lr = 0.01
        self.weight_decay = 0.0
        self.scheduler_step = 5
        self.scheduler_gamma = 0.7
        self.grad_clip = 1.0

        # Early stopping
        self.patience = 5
        self.qae_early_stop_patience = 10

        # Regularizers
        self.use_rl_reg = True
        self.lambda_rl = 0.1

        # QGT specific
        self.freeze_fc_epochs = 3
        self.quantum_lr_mult = 5.0
        self.qk_temp_init = 5.0
        self.mix_init = 1.0

        # Ablations
        self.use_qft = True
        self.use_entanglement = True
        self.use_layernorm = True

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = "results_qgt_6qubit"
        self.save_dir = "saved_models_qgt_6qubit"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# QAE Encoder Circuits
# =============================================================================
def qae_encoder_layer(weights, n_qubits, layer_idx):
    offset = layer_idx * n_qubits * 3
    for q in range(n_qubits):
        qml.RY(weights[offset + q], wires=q)
    for q in range(n_qubits):
        qml.RZ(weights[offset + n_qubits + q], wires=q)
    for q in range(n_qubits):
        qml.CNOT(wires=[q, (q + 1) % n_qubits])
    for q in range(n_qubits):
        qml.RY(weights[offset + 2 * n_qubits + q], wires=q)


def qae_encoder_weights_shape(n_qubits, n_layers):
    return n_layers * n_qubits * 3


@qml.qnode(dev_input, interface="torch", diff_method=DIFF_METHOD)
def qae_trash_expval_circuit(amplitudes, weights, n_layers):
    qml.AmplitudeEmbedding(amplitudes, wires=range(N_QUBITS_INPUT), normalize=True)
    for layer in range(n_layers):
        qae_encoder_layer(weights, N_QUBITS_INPUT, layer)
    return [qml.expval(qml.PauliZ(q)) for q in range(N_QUBITS_LATENT, N_QUBITS_INPUT)]


@qml.qnode(dev_input, interface="torch", diff_method=DIFF_METHOD)
def qae_latent_expval_circuit(amplitudes, weights, n_layers):
    qml.AmplitudeEmbedding(amplitudes, wires=range(N_QUBITS_INPUT), normalize=True)
    for layer in range(n_layers):
        qae_encoder_layer(weights, N_QUBITS_INPUT, layer)
    obs = []
    for q in range(N_QUBITS_LATENT):
        obs.append(qml.expval(qml.PauliX(q)))
    for q in range(N_QUBITS_LATENT):
        obs.append(qml.expval(qml.PauliY(q)))
    for q in range(N_QUBITS_LATENT):
        obs.append(qml.expval(qml.PauliZ(q)))
    return obs


# =============================================================================
# QAE Encoder Module
# =============================================================================
class QAEEncoder(nn.Module):
    def __init__(self, n_layers=2):
        super().__init__()
        self.n_layers = n_layers
        self.n_qubits_input = N_QUBITS_INPUT
        self.n_qubits_latent = N_QUBITS_LATENT
        self.n_qubits_trash = N_QUBITS_TRASH
        self.amp_dim_input = 2 ** N_QUBITS_INPUT
        self.amp_dim_latent = 2 ** N_QUBITS_LATENT
        
        n_params = qae_encoder_weights_shape(N_QUBITS_INPUT, n_layers)
        std = np.sqrt(2.0 / (N_QUBITS_INPUT + n_layers))
        self.weights = nn.Parameter(torch.randn(n_params) * std)
    
    def _prepare_input(self, x):
        batch_size = x.size(0)
        device = x.device
        dtype = x.dtype
        
        if x.size(-1) < self.amp_dim_input:
            padding = torch.zeros(batch_size, self.amp_dim_input - x.size(-1),
                                  device=device, dtype=dtype)
            x_padded = torch.cat([x, padding], dim=-1)
        else:
            x_padded = x[..., :self.amp_dim_input]
        
        x_norm = x_padded / (x_padded.norm(dim=-1, keepdim=True).clamp_min(1e-8))
        return x_norm
    
    def forward(self, x):
        x_norm = self._prepare_input(x)
        batch_size = x_norm.size(0)
        device = x.device
        dtype = x.dtype
        
        outputs = []
        for i in range(batch_size):
            result = qae_latent_expval_circuit(x_norm[i], self.weights, self.n_layers)
            if isinstance(result, list):
                result = torch.stack([
                    r if isinstance(r, torch.Tensor) else torch.tensor(r, device=device)
                    for r in result
                ])
            outputs.append(result)
        
        return torch.stack(outputs, dim=0).to(dtype=dtype)
    
    def get_trash_z_expvals(self, x):
        x_norm = self._prepare_input(x)
        batch_size = x_norm.size(0)
        device = x.device
        dtype = x.dtype
        
        outputs = []
        for i in range(batch_size):
            result = qae_trash_expval_circuit(x_norm[i], self.weights, self.n_layers)
            if isinstance(result, list):
                result = torch.stack([
                    r if isinstance(r, torch.Tensor) else torch.tensor(r, device=device)
                    for r in result
                ])
            outputs.append(result)
        
        return torch.stack(outputs, dim=0).to(dtype=dtype)
    
    def get_compression_quality(self, x):
        z_expvals = self.get_trash_z_expvals(x)
        mean_z = z_expvals.mean(dim=-1)
        quality = (mean_z + 1) / 2
        return quality


# =============================================================================
# QAE Pre-training
# =============================================================================
def train_qae(qae_encoder, train_embeddings, config):
    print("\n" + "=" * 60)
    print("QAE PRE-TRAINING (10 → 6 qubit compression)")
    print("=" * 60)
    print(f"  Learning rate: {config.qae_lr}")
    print(f"  Scheduler: {config.qae_scheduler}")
    print(f"  Layers: {config.qae_n_layers}")
    print(f"  Warmup epochs: {config.qae_warmup_epochs}")
    if config.qae_max_samples:
        print(f"  Max samples: {config.qae_max_samples}")
    print("=" * 60)
    
    device = config.device
    qae_encoder = qae_encoder.to(device)
    
    optimizer = optim.Adam(qae_encoder.parameters(), lr=config.qae_lr)
    
    if config.qae_scheduler == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, 
            patience=config.qae_patience, min_lr=config.qae_min_lr
        )
    elif config.qae_scheduler == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer, T_max=config.qae_epochs, eta_min=config.qae_min_lr
        )
    elif config.qae_scheduler == "step":
        scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    else:
        scheduler = None
    
    if config.qae_max_samples and config.qae_max_samples < train_embeddings.size(0):
        indices = torch.randperm(train_embeddings.size(0))[:config.qae_max_samples]
        train_embeddings = train_embeddings[indices]
        print(f"  Using {train_embeddings.size(0)} samples (subsampled)")
    
    n_samples = train_embeddings.size(0)
    batch_size = min(config.qae_batch_size, n_samples)
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    best_loss = float('inf')
    best_quality = 0.0
    best_state = None
    patience_counter = 0
    history = []
    
    for epoch in range(1, config.qae_epochs + 1):
        qae_encoder.train()
        total_loss = 0.0
        total_quality = 0.0
        
        if epoch <= config.qae_warmup_epochs:
            warmup_lr = config.qae_lr * (epoch / config.qae_warmup_epochs)
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
        
        indices = torch.randperm(n_samples)
        
        pbar = tqdm(range(n_batches), desc=f"QAE Epoch {epoch:02d}", leave=False)
        for batch_idx in pbar:
            start = batch_idx * batch_size
            end = min(start + batch_size, n_samples)
            batch_emb = train_embeddings[indices[start:end]].to(device)
            
            optimizer.zero_grad()
            z_expvals = qae_encoder.get_trash_z_expvals(batch_emb)
            loss = (1 - z_expvals).mean()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(qae_encoder.parameters(), config.grad_clip)
            optimizer.step()
            
            quality = ((z_expvals.mean(dim=-1) + 1) / 2).mean().item()
            total_loss += loss.item()
            total_quality += quality
            
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "quality": f"{quality:.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.6f}"
            })
        
        avg_loss = total_loss / n_batches
        avg_quality = total_quality / n_batches
        current_lr = optimizer.param_groups[0]['lr']
        
        if scheduler is not None and epoch > config.qae_warmup_epochs:
            if config.qae_scheduler == "plateau":
                scheduler.step(avg_loss)
            else:
                scheduler.step()
        
        improved = ""
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_quality = avg_quality
            best_state = {k: v.cpu().clone() for k, v in qae_encoder.state_dict().items()}
            patience_counter = 0
            improved = " *"
        else:
            patience_counter += 1
        
        history.append({
            "epoch": epoch,
            "loss": avg_loss,
            "quality": avg_quality,
            "lr": current_lr
        })
        
        print(f"QAE Epoch {epoch:02d} | Loss={avg_loss:.4f} | Quality={avg_quality:.4f} | LR={current_lr:.6f}{improved}")
        
        if patience_counter >= config.qae_early_stop_patience:
            print(f"\nQAE early stopping at epoch {epoch}")
            break
    
    if best_state is not None:
        qae_encoder.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    
    print(f"\nQAE training complete!")
    print(f"  Best Loss: {best_loss:.4f}")
    print(f"  Best Quality: {best_quality:.4f}")
    
    if best_quality > 0.85:
        quality_str = "EXCELLENT"
    elif best_quality > 0.7:
        quality_str = "GOOD"
    elif best_quality > 0.55:
        quality_str = "MEDIOCRE"
    else:
        quality_str = "POOR"
    print(f"  Compression Rating: {quality_str}")
    
    return qae_encoder, history


# =============================================================================
# 6-Qubit PQC for QGT (FIXED - no AmplitudeEmbedding, use AngleEmbedding)
# =============================================================================
def measure_xyz_6qubit():
    obs = []
    for q in range(N_QUBITS_LATENT):
        obs.append(qml.expval(qml.PauliX(q)))
    for q in range(N_QUBITS_LATENT):
        obs.append(qml.expval(qml.PauliY(q)))
    for q in range(N_QUBITS_LATENT):
        obs.append(qml.expval(qml.PauliZ(q)))
    return obs


@qml.qnode(dev_latent, interface="torch", diff_method=DIFF_METHOD)
def pqc_6qubit_angle(angles, weights):
    """
    6-qubit PQC using AngleEmbedding instead of AmplitudeEmbedding.
    This avoids the NaN normalization issue.
    
    Input: angles (6,) - one angle per qubit
    """
    # Angle embedding - encodes 6 values into 6 qubits
    qml.AngleEmbedding(angles, wires=range(N_QUBITS_LATENT), rotation='Y')
    
    # Variational layers
    for q in range(N_QUBITS_LATENT):
        qml.RX(weights[q, 0], wires=q)
    for q in range(N_QUBITS_LATENT):
        qml.RY(weights[q, 1], wires=q)
    for q in range(N_QUBITS_LATENT):
        qml.RZ(weights[q, 2], wires=q)
    
    # Entanglement
    for q in range(N_QUBITS_LATENT):
        qml.CNOT(wires=[q, (q + 1) % N_QUBITS_LATENT])
    
    # Final rotation
    for q in range(N_QUBITS_LATENT):
        qml.RY(weights[q, 3], wires=q)
    
    # QFT
    qml.QFT(wires=range(N_QUBITS_LATENT))
    
    return measure_xyz_6qubit()


# =============================================================================
# 6-Qubit Q/K Generator (FIXED)
# =============================================================================
class QuantumQKGenerator6Qubit(nn.Module):
    """
    Q/K generator that takes QAE latent output (18 expvals) 
    and produces Q/K vectors via a 6-qubit PQC.
    
    Uses AngleEmbedding (6 angles) instead of AmplitudeEmbedding (64 amplitudes)
    to avoid normalization issues.
    """
    def __init__(self, config):
        super().__init__()
        self.n_qubits = N_QUBITS_LATENT
        self.weights = nn.Parameter(torch.randn(N_QUBITS_LATENT, 4) * 0.01)
        
        # Project 18-dim QAE output to 6-dim for angle encoding
        self.proj = nn.Linear(18, N_QUBITS_LATENT)
    
    def forward(self, qae_output):
        """
        Args:
            qae_output: (batch, 18) - XYZ expvals from QAE encoder
        Returns:
            (batch, 18) - XYZ expvals from 6-qubit PQC
        """
        batch_size = qae_output.size(0)
        device = qae_output.device
        dtype = qae_output.dtype
        
        # Project to 6 angles (scale to [-pi, pi])
        angles = self.proj(qae_output)  # (batch, 6)
        angles = torch.tanh(angles) * np.pi  # Bound to [-pi, pi]
        
        outputs = []
        for i in range(batch_size):
            result = pqc_6qubit_angle(angles[i], self.weights)
            if isinstance(result, list):
                result = torch.stack([
                    r if isinstance(r, torch.Tensor) else torch.tensor(r, device=device)
                    for r in result
                ])
            outputs.append(result)
        
        return torch.stack(outputs, dim=0).to(dtype=dtype)


# =============================================================================
# Utilities
# =============================================================================
class QKTemperature(nn.Module):
    def __init__(self, init_scale=5.0):
        super().__init__()
        self.log_scale = nn.Parameter(torch.tensor(math.log(max(1e-6, init_scale))))
    
    def forward(self):
        return torch.exp(self.log_scale)


def masked_mean_pool(x, batch_idx, pad_mask):
    mask = pad_mask.unsqueeze(-1)
    x = x * mask
    num_graphs = int(batch_idx.max().item()) + 1
    sum_x = scatter_add(x, batch_idx, dim=0, dim_size=num_graphs)
    cnt = scatter_add(mask, batch_idx, dim=0, dim_size=num_graphs).clamp_min(1e-8)
    return sum_x / cnt


# =============================================================================
# 6-Qubit QGT Attention
# =============================================================================
class AttentionConv6Qubit(MessagePassing):
    def __init__(self, config, qae_encoder):
        super().__init__(aggr="add")
        self.config = config
        self.qae_encoder = qae_encoder
        
        for param in self.qae_encoder.parameters():
            param.requires_grad = False
        
        qk_dim = 3 * N_QUBITS_LATENT  # 18
        self.qk_dim = qk_dim
        self.scale = math.sqrt(qk_dim)
        
        self.q_generator = QuantumQKGenerator6Qubit(config)
        self.k_generator = QuantumQKGenerator6Qubit(config)
        self.qk_temp = QKTemperature(config.qk_temp_init)
        self.mix = nn.Parameter(torch.tensor(float(config.mix_init)))
        
        self._alpha = None
        self._alpha_index = None
    
    def forward(self, x, edge_index, pad_mask):
        mask = pad_mask.unsqueeze(-1)
        
        with torch.no_grad():
            qae_latent = self.qae_encoder(x)  # (N, 18)
        
        Q = self.q_generator(qae_latent) * mask
        K = self.k_generator(qae_latent) * mask
        V = x * mask
        
        temp = self.qk_temp().to(x.device, x.dtype)
        Q = Q * temp
        K = K * temp
        
        m = self.propagate(edge_index, q=Q, k=K, v=V)
        out = (x + self.mix * m) * mask
        return out, self._alpha, self._alpha_index
    
    def message(self, q_i, k_j, v_j, index):
        attn_logits = (q_i * k_j).sum(dim=-1) / self.scale
        alpha = pyg_softmax(attn_logits, index)
        self._alpha = alpha
        self._alpha_index = index
        return v_j * alpha.unsqueeze(-1)


# =============================================================================
# 6-Qubit QGT Model
# =============================================================================
class QGT_6Qubit_Model(nn.Module):
    def __init__(self, config, qae_encoder):
        super().__init__()
        self.config = config
        self.qae_encoder = qae_encoder
        
        self.layers = nn.ModuleList([
            AttentionConv6Qubit(config, qae_encoder) 
            for _ in range(config.n_gnn_layers)
        ])
        
        if config.use_layernorm:
            self.norms = nn.ModuleList([
                nn.LayerNorm(config.emb_dim) 
                for _ in range(config.n_gnn_layers)
            ])
        else:
            self.norms = None
        
        self.fc = nn.Linear(config.emb_dim, 2)
    
    def forward(self, batch):
        x = batch.x
        edge_index = batch.edge_index
        pad_mask = batch.pad_mask
        
        alpha, alpha_index = None, None
        for i, conv in enumerate(self.layers):
            x, alpha, alpha_index = conv(x, edge_index, pad_mask)
            if self.norms:
                x = F.relu(self.norms[i](x))
            else:
                x = F.relu(x)
        
        g = masked_mean_pool(x, batch.batch, pad_mask)
        return self.fc(g), (alpha, alpha_index)


# =============================================================================
# Training Utilities
# =============================================================================
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_quantum_params(model):
    total = 0
    for name, param in model.named_parameters():
        if "q_generator.weights" in name or "k_generator.weights" in name:
            total += param.numel()
    return total


def build_optimizer(model, config):
    q_params = []
    other_params = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "q_generator.weights" in n or "k_generator.weights" in n:
            q_params.append(p)
        else:
            other_params.append(p)
    
    param_groups = []
    if other_params:
        param_groups.append({"params": other_params, "lr": config.lr})
    if q_params:
        param_groups.append({"params": q_params, "lr": config.lr * config.quantum_lr_mult})
    
    if not param_groups:
        param_groups = [{"params": model.parameters(), "lr": config.lr}]
    
    return optim.Adam(param_groups, weight_decay=config.weight_decay)


def set_fc_trainable(model, trainable):
    if hasattr(model, "fc"):
        for p in model.fc.parameters():
            p.requires_grad = trainable


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for batch in loader:
        batch = batch.to(device)
        logits, _ = model(batch)
        preds = logits.argmax(dim=1)
        correct += (preds == batch.y).sum().item()
        total += batch.y.size(0)
    return correct / max(1, total)


def rl_regularization(ce_loss, alpha, config):
    if alpha is None or not config.use_rl_reg:
        return torch.tensor(0.0, device=ce_loss.device)
    reward = -ce_loss.detach()
    return -reward * config.lambda_rl


# =============================================================================
# QGT Training
# =============================================================================
def train_qgt(model, train_loader, val_loader, test_loader, config, name="QGT_6Q"):
    device = config.device
    model = model.to(device)
    
    optimizer = build_optimizer(model, config)
    scheduler = StepLR(optimizer, step_size=config.scheduler_step, gamma=config.scheduler_gamma)
    
    best_val, best_epoch = 0.0, 0
    best_state = None
    patience_counter = 0
    history = []
    
    total_params = count_params(model)
    quantum_params = count_quantum_params(model)
    
    print("\n" + "=" * 60)
    print(f"Training {name}")
    print(f"  Total params: {total_params:,}")
    print(f"  Quantum params: {quantum_params}")
    print(f"  Latent qubits: {N_QUBITS_LATENT}")
    print("=" * 60)
    
    for epoch in range(1, config.epochs + 1):
        if epoch <= config.freeze_fc_epochs:
            set_fc_trainable(model, False)
        else:
            set_fc_trainable(model, True)
        
        model.train()
        total_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}", leave=False)
        for batch in pbar:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            
            logits, (alpha, _) = model(batch)
            ce_loss = F.cross_entropy(logits, batch.y)
            rl_loss = rl_regularization(ce_loss, alpha, config)
            loss = ce_loss + rl_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        scheduler.step()
        
        train_acc = evaluate(model, train_loader, device)
        val_acc = evaluate(model, val_loader, device)
        avg_loss = total_loss / len(train_loader)
        
        improved = ""
        if val_acc > best_val:
            best_val = val_acc
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            improved = " *"
        else:
            patience_counter += 1
        
        history.append({"epoch": epoch, "train_acc": train_acc, "val_acc": val_acc, "loss": avg_loss})
        print(f"Epoch {epoch:02d} | Loss={avg_loss:.4f} | Train={train_acc:.4f} | Val={val_acc:.4f}{improved}")
        
        if patience_counter >= config.patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break
    
    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    
    test_acc = evaluate(model, test_loader, device)
    
    print(f"\n{name} Results: Val={best_val:.4f} Test={test_acc:.4f} @ Epoch {best_epoch}")
    
    return {
        "name": name,
        "best_val": best_val,
        "best_test": test_acc,
        "best_epoch": best_epoch,
        "params": total_params,
        "quantum_params": quantum_params,
        "history": history,
    }


# =============================================================================
# Data Loading
# =============================================================================
def load_data(data_path):
    with open(data_path, "rb") as f:
        return pickle.load(f)


def extract_embeddings(data_list, indices):
    embeddings = [data_list[idx].x for idx in indices]
    return torch.cat(embeddings, dim=0)


# =============================================================================
# Main
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(description="QGT with QAE compression (10→6 qubits)")
    
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    
    p.add_argument("--qae_epochs", type=int, default=50)
    p.add_argument("--qae_lr", type=float, default=0.001)
    p.add_argument("--qae_n_layers", type=int, default=2)
    p.add_argument("--qae_batch_size", type=int, default=32)
    p.add_argument("--qae_scheduler", type=str, default="plateau",
                   choices=["plateau", "cosine", "step", "none"])
    p.add_argument("--qae_patience", type=int, default=5)
    p.add_argument("--qae_warmup", type=int, default=3)
    p.add_argument("--qae_max_samples", type=int, default=None)
    
    p.add_argument("--skip_qae", action="store_true")
    p.add_argument("--qae_checkpoint", type=str, default=None)
    
    return p.parse_args()


def main():
    args = parse_args()
    
    print(f"Loading data from {args.data_path}...")
    data = load_data(args.data_path)
    
    data_list = data["data_list"]
    train_idx = data["train_idx"]
    val_idx = data["val_idx"]
    test_idx = data["test_idx"]
    data_config = data.get("config", {})
    
    print(f"Loaded {len(data_list)} samples")
    print(f"Split: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
    
    config = Config()
    config.seed = args.seed
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.lr = args.lr
    config.patience = args.patience
    
    config.qae_epochs = args.qae_epochs
    config.qae_lr = args.qae_lr
    config.qae_n_layers = args.qae_n_layers
    config.qae_batch_size = args.qae_batch_size
    config.qae_scheduler = args.qae_scheduler
    config.qae_patience = args.qae_patience
    config.qae_warmup_epochs = args.qae_warmup
    config.qae_max_samples = args.qae_max_samples
    
    config.emb_dim = int(data_config.get("emb_dim", config.emb_dim))
    
    set_seed(config.seed)
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.save_dir, exist_ok=True)
    
    # Phase 1: QAE Training
    print("\n" + "=" * 70)
    print("PHASE 1: QAE PRE-TRAINING")
    print("=" * 70)
    
    qae_encoder = QAEEncoder(n_layers=config.qae_n_layers)
    
    if args.skip_qae and args.qae_checkpoint:
        print(f"Loading QAE from {args.qae_checkpoint}")
        qae_encoder.load_state_dict(torch.load(args.qae_checkpoint))
        qae_history = []
    else:
        train_embeddings = extract_embeddings(data_list, train_idx)
        print(f"Extracted {train_embeddings.size(0)} embeddings for QAE training")
        qae_encoder, qae_history = train_qae(qae_encoder, train_embeddings, config)
        
        qae_path = os.path.join(config.save_dir, "qae_encoder.pt")
        torch.save(qae_encoder.state_dict(), qae_path)
        print(f"QAE saved to: {qae_path}")
    
    # Phase 2: QGT Training
    print("\n" + "=" * 70)
    print("PHASE 2: QGT TRAINING (6 QUBITS)")
    print("=" * 70)
    
    train_loader = DataLoader([data_list[i] for i in train_idx],
                              batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader([data_list[i] for i in val_idx],
                            batch_size=config.batch_size)
    test_loader = DataLoader([data_list[i] for i in test_idx],
                             batch_size=config.batch_size)
    
    set_seed(config.seed)
    qgt_model = QGT_6Qubit_Model(config, qae_encoder)
    qgt_results = train_qgt(qgt_model, train_loader, val_loader, test_loader, config)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(config.output_dir, f"results_6qubit_{timestamp}.json")
    
    with open(results_file, "w") as f:
        json.dump({
            "qgt_results": qgt_results,
            "qae_history": qae_history,
            "config": {k: str(v) for k, v in vars(config).items()}
        }, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"QAE Compression: 10 qubits → {N_QUBITS_LATENT} qubits")
    print(f"QGT Test Accuracy: {qgt_results['best_test']:.4f}")
    print(f"QGT Parameters: {qgt_results['params']:,} (quantum: {qgt_results['quantum_params']})")


if __name__ == "__main__":
    main()
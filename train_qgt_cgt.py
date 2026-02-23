#!/usr/bin/env python3

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
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

import pennylane as qml

from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax as pyg_softmax
from torch_scatter import scatter_add

# ---------------------------
# Global quantum wires (PennyLane)
# Keep N_QUBITS in sync with config.n_qubits (asserted later)
# ---------------------------
N_QUBITS = 10
try:
    dev = qml.device("lightning.qubit", wires=N_QUBITS)
    QUANTUM_DEVICE = "lightning.qubit"
    DIFF_METHOD = "adjoint"
except Exception as e:
    print(f"Warning: lightning.qubit not available ({e}), using default.qubit)")
    dev = qml.device("default.qubit", wires=N_QUBITS)
    QUANTUM_DEVICE = "default.qubit"
    DIFF_METHOD = "backprop"


# =============================================================================
# Config
# =============================================================================
class Config:
    def __init__(self):
        self.seed = 42

        # Embedding / quantum
        self.emb_dim = 768
        self.n_qubits = 10           # MUST equal N_QUBITS above
        self.amp_dim = 2 ** self.n_qubits  # 1024
        self.n_gnn_layers = 1

        # Training
        self.batch_size = 8
        self.epochs = 50
        self.lr = 0.01
        self.weight_decay = 0.0
        self.scheduler_step = 5
        self.scheduler_gamma = 0.7
        self.grad_clip = 1.0

        # Early stopping
        self.patience = 3

        # Regularizers
        self.use_rl_reg = True
        self.lambda_rl = 0.1
        self.attn_entropy_lambda = 0.01

        # Freeze classifier early
        self.freeze_fc_epochs = 3

        # Quantum / optimizer knobs
        self.quantum_lr_mult = 5.0
        self.qk_temp_mode = "learned"   # "learned" or "fixed"
        self.qk_temp_init = 5.0

        # Residual mixing
        self.mix_init = 1.0

        # Ablations
        self.use_qft = True
        self.use_entanglement = True
        self.entanglement_type = "ring"
        self.use_layernorm = True

        # Debugging
        self.print_every = 20
        self.attn_topk = 5

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Output
        self.output_dir = "results_qgt_cgt"
        self.save_dir = "saved_models_qgt_cgt"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# PennyLane circuits (XYZ measurements)
# =============================================================================
def measure_xyz_all_qubits(n_qubits: int):
    obs = []
    for q in range(n_qubits):
        obs.append(qml.expval(qml.PauliX(q)))
    for q in range(n_qubits):
        obs.append(qml.expval(qml.PauliY(q)))
    for q in range(n_qubits):
        obs.append(qml.expval(qml.PauliZ(q)))
    return obs


@qml.qnode(dev, interface="torch", diff_method=DIFF_METHOD)
def pqc_circuit_full(amplitudes, weights):
    qml.AmplitudeEmbedding(amplitudes, wires=range(N_QUBITS), normalize=True)
    for q in range(N_QUBITS):
        qml.RX(weights[q, 0], wires=q)
    for q in range(N_QUBITS):
        qml.RY(weights[q, 1], wires=q)
    for q in range(N_QUBITS):
        qml.RZ(weights[q, 2], wires=q)
    for q in range(N_QUBITS):
        qml.CNOT(wires=[q, (q + 1) % N_QUBITS])
    for q in range(N_QUBITS):
        qml.RY(weights[q, 3], wires=q)
    qml.QFT(wires=range(N_QUBITS))
    return measure_xyz_all_qubits(N_QUBITS)


@qml.qnode(dev, interface="torch", diff_method=DIFF_METHOD)
def pqc_circuit_no_qft(amplitudes, weights):
    qml.AmplitudeEmbedding(amplitudes, wires=range(N_QUBITS), normalize=True)
    for q in range(N_QUBITS):
        qml.RX(weights[q, 0], wires=q)
    for q in range(N_QUBITS):
        qml.RY(weights[q, 1], wires=q)
    for q in range(N_QUBITS):
        qml.RZ(weights[q, 2], wires=q)
    for q in range(N_QUBITS):
        qml.CNOT(wires=[q, (q + 1) % N_QUBITS])
    for q in range(N_QUBITS):
        qml.RY(weights[q, 3], wires=q)
    return measure_xyz_all_qubits(N_QUBITS)


@qml.qnode(dev, interface="torch", diff_method=DIFF_METHOD)
def pqc_circuit_no_entanglement(amplitudes, weights):
    qml.AmplitudeEmbedding(amplitudes, wires=range(N_QUBITS), normalize=True)
    for q in range(N_QUBITS):
        qml.RX(weights[q, 0], wires=q)
    for q in range(N_QUBITS):
        qml.RY(weights[q, 1], wires=q)
    for q in range(N_QUBITS):
        qml.RZ(weights[q, 2], wires=q)
    for q in range(N_QUBITS):
        qml.RY(weights[q, 3], wires=q)
    qml.QFT(wires=range(N_QUBITS))
    return measure_xyz_all_qubits(N_QUBITS)


@qml.qnode(dev, interface="torch", diff_method=DIFF_METHOD)
def pqc_circuit_linear_entanglement(amplitudes, weights):
    qml.AmplitudeEmbedding(amplitudes, wires=range(N_QUBITS), normalize=True)
    for q in range(N_QUBITS):
        qml.RX(weights[q, 0], wires=q)
    for q in range(N_QUBITS):
        qml.RY(weights[q, 1], wires=q)
    for q in range(N_QUBITS):
        qml.RZ(weights[q, 2], wires=q)
    for q in range(N_QUBITS - 1):
        qml.CNOT(wires=[q, q + 1])
    for q in range(N_QUBITS):
        qml.RY(weights[q, 3], wires=q)
    qml.QFT(wires=range(N_QUBITS))
    return measure_xyz_all_qubits(N_QUBITS)


@qml.qnode(dev, interface="torch", diff_method=DIFF_METHOD)
def pqc_circuit_no_entanglement_no_qft(amplitudes, weights):
    """Minimal circuit: only rotation gates, no entanglement, no QFT"""
    qml.AmplitudeEmbedding(amplitudes, wires=range(N_QUBITS), normalize=True)
    for q in range(N_QUBITS):
        qml.RX(weights[q, 0], wires=q)
    for q in range(N_QUBITS):
        qml.RY(weights[q, 1], wires=q)
    for q in range(N_QUBITS):
        qml.RZ(weights[q, 2], wires=q)
    for q in range(N_QUBITS):
        qml.RY(weights[q, 3], wires=q)
    return measure_xyz_all_qubits(N_QUBITS)


@qml.qnode(dev, interface="torch", diff_method=DIFF_METHOD)
def pqc_circuit_linear_no_qft(amplitudes, weights):
    """Linear entanglement without QFT"""
    qml.AmplitudeEmbedding(amplitudes, wires=range(N_QUBITS), normalize=True)
    for q in range(N_QUBITS):
        qml.RX(weights[q, 0], wires=q)
    for q in range(N_QUBITS):
        qml.RY(weights[q, 1], wires=q)
    for q in range(N_QUBITS):
        qml.RZ(weights[q, 2], wires=q)
    for q in range(N_QUBITS - 1):
        qml.CNOT(wires=[q, q + 1])
    for q in range(N_QUBITS):
        qml.RY(weights[q, 3], wires=q)
    return measure_xyz_all_qubits(N_QUBITS)


def get_pqc_circuit(config: Config):
    """Select PQC circuit based on ablation settings"""
    if not config.use_entanglement and not config.use_qft:
        return pqc_circuit_no_entanglement_no_qft
    elif not config.use_entanglement:
        return pqc_circuit_no_entanglement
    elif config.entanglement_type == "linear" and not config.use_qft:
        return pqc_circuit_linear_no_qft
    elif config.entanglement_type == "linear":
        return pqc_circuit_linear_entanglement
    elif not config.use_qft:
        return pqc_circuit_no_qft
    else:
        return pqc_circuit_full


# =============================================================================
# Q/K Generators
# - QGT: QuantumQKGenerator (PQC)
# - CGT: ClassicalQKGenerator (linear projection)
# Both output dim = 3 * n_qubits to be comparable
# =============================================================================
class QuantumQKGenerator(nn.Module):
    def __init__(self, n_qubits=10, config: Config = None):
        super().__init__()
        self.n_qubits = n_qubits
        self.amp_dim = 2 ** n_qubits
        self.config = config
        self.circuit = get_pqc_circuit(config) if config is not None else pqc_circuit_full
        # PQC weights: (n_qubits, 4)
        self.weights = nn.Parameter(torch.randn(n_qubits, 4) * 0.01)

    def forward(self, x):
        # x : (N, emb_dim)
        batch_size = x.size(0)
        device = x.device
        dtype = x.dtype

        if x.size(-1) < self.amp_dim:
            padding = torch.zeros(batch_size, self.amp_dim - x.size(-1), device=device, dtype=dtype)
            x_padded = torch.cat([x, padding], dim=-1)
        else:
            x_padded = x[..., : self.amp_dim]

        x_norm = x_padded / (x_padded.norm(dim=-1, keepdim=True).clamp_min(1e-8))

        outputs = []
        for i in range(batch_size):
            result = self.circuit(x_norm[i], self.weights)
            if isinstance(result, list):
                result = torch.stack([
                    r if isinstance(r, torch.Tensor) else torch.tensor(r, device=device)
                    for r in result
                ])
            outputs.append(result)
        out = torch.stack(outputs, dim=0)
        return out.to(dtype=dtype)  # (batch, 3*n_qubits)


class ClassicalQKGenerator(nn.Module):
    def __init__(self, emb_dim, qk_dim):
        super().__init__()
        self.q_proj = nn.Linear(emb_dim, qk_dim, bias=False)
        self.k_proj = nn.Linear(emb_dim, qk_dim, bias=False)

    def forward(self, x):
        Q = self.q_proj(x)
        K = self.k_proj(x)
        return Q, K


# =============================================================================
# Helpers: attention stats, grad norms, temperature
# =============================================================================
def attention_stats(alpha, dst_index, edge_index=None, topk=5, eps=1e-12):
    if alpha is None or dst_index is None or alpha.numel() == 0:
        return None

    a = alpha.clamp_min(eps)
    ones = torch.ones_like(dst_index, dtype=a.dtype)
    deg = scatter_add(ones, dst_index, dim=0)
    deg_mean = float(deg.mean().item())
    frac_deg1 = float((deg == 1).float().mean().item())

    ent_edge = -(a * a.log())
    ent_per_dst = scatter_add(ent_edge, dst_index, dim=0)

    mask_nt = deg > 1
    ent_mean_nt = float(ent_per_dst[mask_nt].mean().item()) if mask_nt.any() else float("nan")

    rowmax_mean_nt = float("nan")
    try:
        from torch_scatter import scatter_max
        rowmax_per_dst, _ = scatter_max(a, dst_index, dim=0)
        if mask_nt.any():
            rowmax_mean_nt = float(rowmax_per_dst[mask_nt].mean().item())
    except Exception:
        pass

    # Select top edges only among non-trivial dest rows and exclude self-loops if possible
    if edge_index is not None:
        src = edge_index[0]
        dst = edge_index[1]
        deg_per_edge = deg[dst.long()]
        keep = deg_per_edge > 1
        # exclude self-loops
        keep = keep & (src != dst)
        if keep.any():
            a2 = a[keep]
            src2 = src[keep]
            dst2 = dst[keep]
            k = min(int(topk), int(a2.numel()))
            vals, idx = torch.topk(a2, k=k, largest=True)
            top_edges = [(int(src2[i].item()), int(dst2[i].item()), float(vals[j].item()))
                         for j, i in enumerate(idx)]
        else:
            top_edges = []
    else:
        top_edges = []

    return {
        "deg_mean": deg_mean,
        "frac_deg1": frac_deg1,
        "ent_mean_nt": ent_mean_nt,
        "rowmax_mean_nt": rowmax_mean_nt,
        "top_edges": top_edges,
    }


def grad_norm_named(model, name_filter=None):
    total = 0.0
    for n, p in model.named_parameters():
        if p.grad is None:
            continue
        if name_filter is not None and name_filter not in n:
            continue
        total += float(p.grad.detach().pow(2).sum().item())
    return math.sqrt(total)


def param_groups_grad_norms(model):
    g = {}
    g["quantum_Q"] = grad_norm_named(model, "q_generator.weights")
    g["quantum_K"] = grad_norm_named(model, "k_generator.weights")
    g["layernorm"] = grad_norm_named(model, "norms")
    g["classifier"] = grad_norm_named(model, "fc")
    g["all"] = grad_norm_named(model, None)
    return g


class QKTemperature(nn.Module):
    def __init__(self, init_scale=5.0, mode="learned"):
        super().__init__()
        self.mode = mode
        if mode == "learned":
            init_log = math.log(max(1e-6, float(init_scale)))
            self.log_scale = nn.Parameter(torch.tensor(init_log, dtype=torch.float32))
        elif mode == "fixed":
            self.register_buffer("fixed_scale", torch.tensor(float(init_scale), dtype=torch.float32))
        else:
            raise ValueError("qk_temp_mode must be 'learned' or 'fixed'")

    def forward(self):
        if self.mode == "learned":
            return torch.exp(self.log_scale)
        return self.fixed_scale


# =============================================================================
# Attention Convolution (shared structure for QGT & CGT)
# - QGT uses QuantumQKGenerator
# - CGT uses ClassicalQKGenerator but same rest of logic
# =============================================================================
class AttentionConv(MessagePassing):
    def __init__(self, config: Config, use_quantum: bool):
        super().__init__(aggr="add")
        self.config = config
        self.use_quantum = use_quantum

        qk_dim = 3 * config.n_qubits
        self.qk_dim = qk_dim
        self.scale = math.sqrt(qk_dim)

        if use_quantum:
            self.q_generator = QuantumQKGenerator(config.n_qubits, config=config)
            self.k_generator = QuantumQKGenerator(config.n_qubits, config=config)
        else:
            # classical projections (one module for Q/K each)
            self.classical_q = nn.Linear(config.emb_dim, qk_dim, bias=False)
            self.classical_k = nn.Linear(config.emb_dim, qk_dim, bias=False)

        # temperature / scale
        self.qk_temp = QKTemperature(init_scale=config.qk_temp_init, mode=config.qk_temp_mode)

        # learnable residual mix (scalar)
        self.mix = nn.Parameter(torch.tensor(float(config.mix_init)))

        # for diagnostics
        self._alpha = None
        self._alpha_index = None

    def forward(self, x, edge_index, pad_mask):
        mask = pad_mask.unsqueeze(-1)
        if self.use_quantum:
            Q = self.q_generator(x) * mask
            K = self.k_generator(x) * mask
        else:
            Q = self.classical_q(x) * mask
            K = self.classical_k(x) * mask
        V = x * mask

        temp = self.qk_temp().to(x.device, x.dtype)
        Q = Q * temp
        K = K * temp

        m = self.propagate(edge_index, q=Q, k=K, v=V)  # aggregated messages
        out = (x + (self.mix * m)) * mask
        return out, self._alpha, self._alpha_index

    def message(self, q_i, k_j, v_j, index):
        attn_logits = (q_i * k_j).sum(dim=-1) / self.scale
        alpha = pyg_softmax(attn_logits, index)
        self._alpha = alpha
        self._alpha_index = index
        return v_j * alpha.unsqueeze(-1)


# =============================================================================
# QGT and CGT Models
# =============================================================================
def masked_mean_pool(x, batch_idx, pad_mask):
    mask = pad_mask.unsqueeze(-1)
    x = x * mask
    num_graphs = int(batch_idx.max().item()) + 1
    sum_x = scatter_add(x, batch_idx, dim=0, dim_size=num_graphs)
    cnt = scatter_add(mask, batch_idx, dim=0, dim_size=num_graphs).clamp_min(1e-8)
    return sum_x / cnt


class QGT_Model(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([AttentionConv(config, use_quantum=True) for _ in range(config.n_gnn_layers)])
        if config.use_layernorm:
            self.norms = nn.ModuleList([nn.LayerNorm(config.emb_dim) for _ in range(config.n_gnn_layers)])
        else:
            self.norms = None
        self.fc = nn.Linear(config.emb_dim, 2)

    def forward(self, batch):
        x = batch.x
        edge_index = batch.edge_index
        pad_mask = batch.pad_mask

        alpha = None
        alpha_index = None
        for i, conv in enumerate(self.layers):
            x, alpha, alpha_index = conv(x, edge_index, pad_mask)
            if self.norms is not None:
                x = F.relu(self.norms[i](x))
            else:
                x = F.relu(x)

        g = masked_mean_pool(x, batch.batch, pad_mask)
        return self.fc(g), (alpha, alpha_index)


class CGT_Model(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([AttentionConv(config, use_quantum=False) for _ in range(config.n_gnn_layers)])
        if config.use_layernorm:
            self.norms = nn.ModuleList([nn.LayerNorm(config.emb_dim) for _ in range(config.n_gnn_layers)])
        else:
            self.norms = None
        self.fc = nn.Linear(config.emb_dim, 2)

    def forward(self, batch):
        x = batch.x
        edge_index = batch.edge_index
        pad_mask = batch.pad_mask

        alpha = None
        alpha_index = None
        for i, conv in enumerate(self.layers):
            x, alpha, alpha_index = conv(x, edge_index, pad_mask)
            if self.norms is not None:
                x = F.relu(self.norms[i](x))
            else:
                x = F.relu(x)

        g = masked_mean_pool(x, batch.batch, pad_mask)
        return self.fc(g), (alpha, alpha_index)


# =============================================================================
# Regularization & utilities
# =============================================================================
def rl_regularization(ce_loss, alpha, config: Config):
    if alpha is None or not config.use_rl_reg:
        return torch.tensor(0.0, device=ce_loss.device)
    reward = -ce_loss.detach()
    reg = -reward * config.lambda_rl
    return reg


def attn_entropy_regularizer(alpha, alpha_index, config: Config):
    """Encourage higher entropy on deg>1 rows (small positive lambda)."""
    if alpha is None or alpha_index is None:
        return torch.tensor(0.0)
    a = alpha.clamp_min(1e-12)
    # compute entropy per dst
    ent_edge = -(a * a.log())
    ent_per_dst = scatter_add(ent_edge, alpha_index, dim=0)
    # compute deg per dst
    ones = torch.ones_like(alpha_index, dtype=a.dtype)
    deg = scatter_add(ones, alpha_index, dim=0)
    mask = deg > 1
    if not mask.any():
        return torch.tensor(0.0, device=a.device)
    ent_mean = ent_per_dst[mask].mean()
    # maximize entropy -> subtract scaled entropy (we add lambda * (-ent_mean) to loss)
    return -config.attn_entropy_lambda * ent_mean


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_quantum_params(model):
    quantum_params = 0
    for name, param in model.named_parameters():
        if "q_generator.weights" in name or "k_generator.weights" in name:
            quantum_params += param.numel()
    return quantum_params


# =============================================================================
# Optimizer builder (robust)
# =============================================================================
def build_optimizer(model: nn.Module, config: Config):
    q_params = []
    other_params = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # identify PQC params by substring (QuantumQKGenerator.weights)
        if "q_generator.weights" in n or "k_generator.weights" in n:
            q_params.append(p)
        else:
            other_params.append(p)

    param_groups = []
    if len(other_params) > 0:
        param_groups.append({"params": other_params, "lr": config.lr})
    if len(q_params) > 0:
        param_groups.append({"params": q_params, "lr": config.lr * config.quantum_lr_mult})

    if len(param_groups) == 0:
        req = [p for p in model.parameters() if p.requires_grad]
        if len(req) > 0:
            param_groups = [{"params": req, "lr": config.lr}]
            print(f"[build_optimizer] Warning: fell back to {len(req)} requires_grad params.", flush=True)
        else:
            param_groups = [{"params": list(model.parameters()), "lr": config.lr}]
            print("[build_optimizer] WARNING: no params require grad; optimizer will include all params (may be frozen).", flush=True)

    optimizer = optim.Adam(param_groups, weight_decay=config.weight_decay)
    return optimizer


def set_fc_trainable(model: nn.Module, trainable: bool):
    if hasattr(model, "fc"):
        for p in model.fc.parameters():
            p.requires_grad = trainable


# =============================================================================
# Training & Evaluation
# =============================================================================
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


def train_model(model, train_loader, val_loader, test_loader, config: Config, name: str):
    device = config.device
    model = model.to(device)

    optimizer = build_optimizer(model, config)
    scheduler = StepLR(optimizer, step_size=config.scheduler_step, gamma=config.scheduler_gamma)

    best_val, best_epoch = 0.0, 0
    best_state = None
    patience_counter = 0
    history = []

    total_params = count_params(model)
    quantum_params = count_quantum_params(model) if isinstance(model, QGT_Model) else 0

    # Build ablation description for logging
    ablation_desc = []
    if not config.use_qft:
        ablation_desc.append("no_qft")
    if not config.use_entanglement:
        ablation_desc.append("no_entanglement")
    elif config.entanglement_type == "linear":
        ablation_desc.append("linear_entanglement")
    if not config.use_rl_reg:
        ablation_desc.append("no_rl_reg")
    if not config.use_layernorm:
        ablation_desc.append("no_layernorm")
    ablation_str = ", ".join(ablation_desc) if ablation_desc else "full"

    print("\n" + "=" * 60)
    print(f"Training {name}")
    print(f"  Ablation: {ablation_str}")
    print(f"  Total params: {total_params:,}")
    if quantum_params > 0:
        print(f"  Quantum params: {quantum_params}")
        print(f"  Quantum LR mult: {config.quantum_lr_mult}x")
    print(f"  Early Stopping: patience={config.patience}")
    print(f"  Debug: print_every={config.print_every} attn_topk={config.attn_topk}")
    print("=" * 60)

    prev_grad_stats = None

    for epoch in range(1, config.epochs + 1):
        # freeze fc only for QGT and CGT (both should have name in ["QGT","CGT"])
        if epoch <= config.freeze_fc_epochs and name.upper() in ("QGT", "CGT"):
            set_fc_trainable(model, False)
        else:
            set_fc_trainable(model, True)

        model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}", leave=False)
        for step, batch in enumerate(pbar, start=1):
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)

            logits, (alpha, alpha_index) = model(batch)
            ce_loss = F.cross_entropy(logits, batch.y)
            rl_loss = rl_regularization(ce_loss, alpha, config)
            ent_reg = attn_entropy_regularizer(alpha, alpha_index, config)
            loss = ce_loss + rl_loss + ent_reg

            # backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

            grad_stats = param_groups_grad_norms(model)
            delta_q = delta_k = None
            if prev_grad_stats is not None:
                delta_q = abs(grad_stats["quantum_Q"] - prev_grad_stats["quantum_Q"])
                delta_k = abs(grad_stats["quantum_K"] - prev_grad_stats["quantum_K"])

            if config.print_every > 0 and (step % config.print_every) == 0:
                att = attention_stats(alpha, alpha_index, edge_index=batch.edge_index, topk=config.attn_topk)
                if att is not None:
                    print(
                        f"[{name}] ep={epoch:02d} step={step:04d} "
                        f"attn(deg_mean={att['deg_mean']:.2f}, frac_deg1={att['frac_deg1']:.2f}, "
                        f"ent_nt={att['ent_mean_nt']:.3f}, rowmax_nt={att['rowmax_mean_nt']:.3f}, top={att['top_edges']}) "
                        f"grad(Q={grad_stats['quantum_Q']:.2e}, K={grad_stats['quantum_K']:.2e}, "
                        f"fc={grad_stats['classifier']:.2e}, all={grad_stats['all']:.2e}) "
                        + (f"Δgrad(Q={delta_q:.2e}, K={delta_k:.2e})" if delta_q is not None else ""),
                        flush=True
                    )
                else:
                    print(
                        f"[{name}] ep={epoch:02d} step={step:04d} "
                        f"grad(Q={grad_stats['quantum_Q']:.2e}, K={grad_stats['quantum_K']:.2e}, "
                        f"fc={grad_stats['classifier']:.2e}, all={grad_stats['all']:.2e}) "
                        + (f"Δgrad(Q={delta_q:.2e}, K={delta_k:.2e})" if delta_q is not None else ""),
                        flush=True
                    )

            prev_grad_stats = grad_stats

            optimizer.step()
            total_loss += float(loss.item())
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()

        train_acc = evaluate(model, train_loader, device)
        val_acc = evaluate(model, val_loader, device)
        avg_loss = total_loss / max(1, len(train_loader))

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

        print(f"Epoch {epoch:02d} | Loss={avg_loss:.4f} | Train={train_acc:.4f} Val={val_acc:.4f}{improved}")

        if patience_counter >= config.patience:
            print(f"\nEarly stopping at epoch {epoch} (patience={config.patience})")
            break

    if best_state is not None:
        model.load_state_dict({k: v.to(config.device) for k, v in best_state.items()})

    test_acc = evaluate(model, test_loader, device)

    os.makedirs(config.save_dir, exist_ok=True)
    
    # Build ablation suffix for filename
    ablation_parts = []
    if not config.use_qft:
        ablation_parts.append("noqft")
    if not config.use_entanglement:
        ablation_parts.append("noent")
    elif config.entanglement_type == "linear":
        ablation_parts.append("linear")
    if not config.use_rl_reg:
        ablation_parts.append("norl")
    if not config.use_layernorm:
        ablation_parts.append("noln")
    ablation_suffix = "_" + "_".join(ablation_parts) if ablation_parts else "_full"
    
    model_path = os.path.join(config.save_dir, f"{name.lower()}{ablation_suffix}_best.pt")
    torch.save({
        "model_state_dict": best_state,
        "config": vars(config),
        "best_val": best_val,
        "best_epoch": best_epoch,
        "test_acc": test_acc,
        "ablation": ablation_str,
    }, model_path)

    print(f"Best model saved to: {model_path}")
    print(f"\n{name} Results: Val={best_val:.4f} Test={test_acc:.4f} @ Epoch {best_epoch}")

    return {
        "name": name,
        "ablation": ablation_str,
        "best_val": best_val,
        "best_test": test_acc,
        "best_epoch": best_epoch,
        "params": total_params,
        "quantum_params": quantum_params,
        "history": history,
        "model_path": model_path,
    }


# =============================================================================
# Data loader helper
# =============================================================================
def load_preprocessed_data(data_path):
    with open(data_path, "rb") as f:
        return pickle.load(f)


# =============================================================================
# Main
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(description="QGT vs CGT training (parity)")

    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--print_every", type=int, default=20)
    p.add_argument("--attn_topk", type=int, default=5)

    p.add_argument("--freeze_fc_epochs", type=int, default=3)
    p.add_argument("--quantum_lr_mult", type=float, default=5.0)
    p.add_argument("--qk_temp_mode", type=str, choices=["learned", "fixed"], default="learned")
    p.add_argument("--qk_temp_init", type=float, default=5.0)

    # Ablation study arguments
    p.add_argument("--no_qft", action="store_true", help="Disable QFT layer in quantum circuit")
    p.add_argument("--no_rl_reg", action="store_true", help="Disable RL regularization")
    p.add_argument("--no_entanglement", action="store_true", help="Disable CNOT entanglement gates")
    p.add_argument("--linear_entanglement", action="store_true", help="Use linear CNOT pattern instead of ring")
    p.add_argument("--no_layernorm", action="store_true", help="Disable LayerNorm after attention")
    p.add_argument("--lambda_rl", type=float, default=0.1, help="RL regularization strength")
    p.add_argument("--attn_entropy_lambda", type=float, default=0.01, help="Attention entropy regularization")
    
    # Run only specific model
    p.add_argument("--qgt_only", action="store_true", help="Run only QGT (skip CGT)")
    p.add_argument("--cgt_only", action="store_true", help="Run only CGT (skip QGT)")

    return p.parse_args()


def main():
    args = parse_args()
    print(f"Loading data from {args.data_path}...")
    data = load_preprocessed_data(args.data_path)

    data_list = data["data_list"]
    train_idx = data["train_idx"]
    val_idx = data["val_idx"]
    test_idx = data["test_idx"]
    data_config = data.get("config", {})

    print(f"Loaded {len(data_list)} samples")
    print(f"Embedding: {data_config.get('embedding_type', 'unknown')}")
    print(f"Graph: {data_config.get('graph_type', 'unknown')}")
    print(f"Split: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")

    config = Config()
    config.seed = args.seed
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.lr = args.lr
    config.patience = args.patience
    config.freeze_fc_epochs = args.freeze_fc_epochs
    config.quantum_lr_mult = args.quantum_lr_mult
    config.qk_temp_mode = args.qk_temp_mode
    config.qk_temp_init = args.qk_temp_init
    config.print_every = args.print_every
    config.attn_topk = args.attn_topk

    # Ablation settings
    config.use_qft = not args.no_qft
    config.use_rl_reg = not args.no_rl_reg
    config.use_entanglement = not args.no_entanglement
    config.use_layernorm = not args.no_layernorm
    config.lambda_rl = args.lambda_rl
    config.attn_entropy_lambda = args.attn_entropy_lambda
    
    if args.linear_entanglement:
        config.entanglement_type = "linear"
    else:
        config.entanglement_type = "ring"

    # use saved data config if present
    config.emb_dim = int(data_config.get("emb_dim", config.emb_dim))
    config.n_qubits = int(data_config.get("n_qubits", config.n_qubits))
    config.amp_dim = int(data_config.get("amp_dim", config.amp_dim))

    # sanity: ensure PennyLane wires match config.n_qubits
    assert config.n_qubits == N_QUBITS, f"N_QUBITS ({N_QUBITS}) must equal config.n_qubits ({config.n_qubits}). Edit script top to change wires."

    # Print ablation configuration
    print("\n" + "=" * 60)
    print("ABLATION CONFIGURATION")
    print("=" * 60)
    print(f"  QFT Layer:         {'ENABLED' if config.use_qft else 'DISABLED'}")
    print(f"  Entanglement:      {'DISABLED' if not config.use_entanglement else config.entanglement_type.upper()}")
    print(f"  RL Regularization: {'ENABLED (λ=' + str(config.lambda_rl) + ')' if config.use_rl_reg else 'DISABLED'}")
    print(f"  LayerNorm:         {'ENABLED' if config.use_layernorm else 'DISABLED'}")
    print(f"  Attn Entropy λ:    {config.attn_entropy_lambda}")
    print("=" * 60)

    set_seed(config.seed)
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.save_dir, exist_ok=True)

    train_loader = DataLoader([data_list[i] for i in train_idx], batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader([data_list[i] for i in val_idx], batch_size=config.batch_size)
    test_loader = DataLoader([data_list[i] for i in test_idx], batch_size=config.batch_size)

    # instantiate models
    qgt = QGT_Model(config)
    cgt = CGT_Model(config)

    print("\nPARAMETER COUNT")
    print(f"QGT: {count_params(qgt):,} params (quantum: {count_quantum_params(qgt)})")
    print(f"CGT: {count_params(cgt):,} params")

    results = {}

    if not args.cgt_only:
        set_seed(config.seed)
        results["QGT"] = train_model(QGT_Model(config), train_loader, val_loader, test_loader, config, "QGT")

    if not args.qgt_only:
        set_seed(config.seed)
        results["CGT"] = train_model(CGT_Model(config), train_loader, val_loader, test_loader, config, "CGT")

    # Build ablation suffix for output filename
    ablation_parts = []
    if not config.use_qft:
        ablation_parts.append("noqft")
    if not config.use_entanglement:
        ablation_parts.append("noent")
    elif config.entanglement_type == "linear":
        ablation_parts.append("linear")
    if not config.use_rl_reg:
        ablation_parts.append("norl")
    if not config.use_layernorm:
        ablation_parts.append("noln")
    ablation_file_suffix = "_" + "_".join(ablation_parts) if ablation_parts else ""

    # final save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(config.output_dir, f"results_qgt_cgt{ablation_file_suffix}_{timestamp}.json")
    save_data = {"results": results, "config": vars(config)}
    with open(out_file, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nResults saved to: {out_file}")

    print("\nSUMMARY")
    print(f"{'Model':<6} {'Ablation':<25} {'Params':>10} {'Quantum':>10} {'Val':>8} {'Test':>8}")
    for name, r in results.items():
        qparams = r.get("quantum_params", 0)
        ablation = r.get("ablation", "full")
        print(f"{name:<6} {ablation:<25} {r['params']:>10,} {qparams:>10} {r['best_val']:>8.4f} {r['best_test']:>8.4f}")


if __name__ == "__main__":
    main()
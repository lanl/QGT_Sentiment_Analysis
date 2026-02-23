#!/usr/bin/env python3
"""
Sample Efficiency Analysis: QGT with varying training data.

Usage:
    python sample_efficiency_qgt.py --data_path data.pkl --fractions 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
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
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

import pennylane as qml

from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax as pyg_softmax
from torch_scatter import scatter_add


# =============================================================================
# Quantum Device Setup
# =============================================================================
N_QUBITS = 10
try:
    dev = qml.device("lightning.qubit", wires=N_QUBITS)
    QUANTUM_DEVICE = "lightning.qubit"
    DIFF_METHOD = "adjoint"
except Exception as e:
    print(f"Warning: lightning.qubit not available ({e}), using default.qubit")
    dev = qml.device("default.qubit", wires=N_QUBITS)
    QUANTUM_DEVICE = "default.qubit"
    DIFF_METHOD = "backprop"

print(f"Quantum device: {QUANTUM_DEVICE} ({DIFF_METHOD})")


# =============================================================================
# Config
# =============================================================================
class Config:
    def __init__(self):
        self.seed = 42
        self.emb_dim = 768
        self.n_qubits = 10
        self.amp_dim = 2 ** self.n_qubits
        self.n_gnn_layers = 1

        self.batch_size = 32
        self.epochs = 50
        self.lr = 0.01
        self.weight_decay = 0.0
        self.scheduler_step = 5
        self.scheduler_gamma = 0.7
        self.grad_clip = 1.0
        self.patience = 5

        self.use_rl_reg = True
        self.lambda_rl = 0.1

        self.freeze_fc_epochs = 3
        self.quantum_lr_mult = 5.0
        self.qk_temp_init = 5.0
        self.mix_init = 1.0

        self.use_qft = True
        self.use_entanglement = True
        self.use_layernorm = False

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = "results_sample_efficiency"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# PennyLane Circuit
# =============================================================================
def measure_xyz_all_qubits(n_qubits):
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


# =============================================================================
# Quantum Q/K Generator
# =============================================================================
class QuantumQKGenerator(nn.Module):
    def __init__(self, n_qubits=10):
        super().__init__()
        self.n_qubits = n_qubits
        self.amp_dim = 2 ** n_qubits
        self.weights = nn.Parameter(torch.randn(n_qubits, 4) * 0.01)

    def forward(self, x):
        batch_size = x.size(0)
        device = x.device
        dtype = x.dtype

        if x.size(-1) < self.amp_dim:
            padding = torch.zeros(batch_size, self.amp_dim - x.size(-1), device=device, dtype=dtype)
            x_padded = torch.cat([x, padding], dim=-1)
        else:
            x_padded = x[..., :self.amp_dim]

        x_norm = x_padded / (x_padded.norm(dim=-1, keepdim=True).clamp_min(1e-8))

        outputs = []
        for i in range(batch_size):
            result = pqc_circuit_full(x_norm[i], self.weights)
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


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_quantum_params(model):
    total = 0
    for name, param in model.named_parameters():
        if "q_generator.weights" in name or "k_generator.weights" in name:
            total += param.numel()
    return total


# =============================================================================
# QGT Attention Conv
# =============================================================================
class AttentionConvQuantum(MessagePassing):
    def __init__(self, config):
        super().__init__(aggr="add")
        self.config = config

        qk_dim = 3 * config.n_qubits  # 30
        self.qk_dim = qk_dim
        self.scale = math.sqrt(qk_dim)

        self.q_generator = QuantumQKGenerator(config.n_qubits)
        self.k_generator = QuantumQKGenerator(config.n_qubits)

        self.qk_temp = QKTemperature(init_scale=config.qk_temp_init)
        self.mix = nn.Parameter(torch.tensor(float(config.mix_init)))

        self._alpha = None
        self._alpha_index = None

    def forward(self, x, edge_index, pad_mask):
        mask = pad_mask.unsqueeze(-1)

        Q = self.q_generator(x) * mask
        K = self.k_generator(x) * mask
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
# QGT Model
# =============================================================================
class QGT_Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.layers = nn.ModuleList([
            AttentionConvQuantum(config) for _ in range(config.n_gnn_layers)
        ])

        if config.use_layernorm:
            self.norms = nn.ModuleList([
                nn.LayerNorm(config.emb_dim) for _ in range(config.n_gnn_layers)
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
# Training
# =============================================================================
def rl_regularization(ce_loss, alpha, config):
    if alpha is None or not config.use_rl_reg:
        return torch.tensor(0.0, device=ce_loss.device)
    reward = -ce_loss.detach()
    return -reward * config.lambda_rl


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


def train_model(model, train_loader, val_loader, test_loader, config, name):
    device = config.device
    model = model.to(device)

    optimizer = build_optimizer(model, config)
    scheduler = StepLR(optimizer, step_size=config.scheduler_step, gamma=config.scheduler_gamma)

    best_val, best_epoch = 0.0, 0
    best_state = None
    patience_counter = 0

    for epoch in range(1, config.epochs + 1):
        if epoch <= config.freeze_fc_epochs:
            set_fc_trainable(model, False)
        else:
            set_fc_trainable(model, True)

        model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"{name} Epoch {epoch:02d}", leave=False)
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

        val_acc = evaluate(model, val_loader, device)

        if val_acc > best_val:
            best_val = val_acc
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config.patience:
            break

    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    test_acc = evaluate(model, test_loader, device)

    return {
        "best_val": best_val,
        "best_test": test_acc,
        "best_epoch": best_epoch,
        "params": count_params(model),
        "quantum_params": count_quantum_params(model),
    }


# =============================================================================
# Sample Efficiency Experiment
# =============================================================================
def run_sample_efficiency(data_path, fractions, config):
    print(f"\nLoading data from {data_path}...")
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    data_list = data["data_list"]
    train_idx = data["train_idx"]
    val_idx = data["val_idx"]
    test_idx = data["test_idx"]
    data_config = data.get("config", {})

    config.emb_dim = int(data_config.get("emb_dim", config.emb_dim))

    model_temp = QGT_Model(config)
    print(f"Total training samples: {len(train_idx)}")
    print(f"Model: QGT (Quantum Q/K, {config.n_qubits} qubits)")
    print(f"Total Params: {count_params(model_temp):,}")
    print(f"Quantum Params: {count_quantum_params(model_temp)}")

    val_loader = DataLoader([data_list[i] for i in val_idx], batch_size=config.batch_size)
    test_loader = DataLoader([data_list[i] for i in test_idx], batch_size=config.batch_size)

    results = {}

    print("\n" + "=" * 50)
    print("Running Sample Efficiency Experiment...")
    print("=" * 50)

    for frac in fractions:
        set_seed(config.seed)
        n_samples = max(1, int(len(train_idx) * frac))
        sampled_train_idx = random.sample(train_idx, n_samples)

        train_loader = DataLoader(
            [data_list[i] for i in sampled_train_idx],
            batch_size=config.batch_size,
            shuffle=True
        )

        set_seed(config.seed)
        model = QGT_Model(config)

        result = train_model(model, train_loader, val_loader, test_loader, config, f"QGT_{frac*100:.0f}%")
        result["fraction"] = frac
        result["n_samples"] = n_samples
        results[frac] = result

    # Print results
    print("\n" + "=" * 50)
    print("SAMPLE EFFICIENCY RESULTS (QGT)")
    print("=" * 50)
    print(f"{'Fraction':<12} {'Samples':<10} {'Test Acc':<12}")
    print("-" * 50)

    for frac in sorted(results.keys()):
        r = results[frac]
        print(f"{frac*100:>6.0f}%      {r['n_samples']:<10} {r['best_test']*100:.2f}%")

    print("=" * 50)

    # Save results
    os.makedirs(config.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(config.output_dir, f"sample_efficiency_qgt_{timestamp}.json")

    with open(out_file, "w") as f:
        json.dump({
            "results": {str(k): v for k, v in results.items()},
            "config": {k: str(v) for k, v in vars(config).items()}
        }, f, indent=2)

    print(f"\nResults saved to: {out_file}")

    return results


# =============================================================================
# Main
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(description="Sample Efficiency: QGT")

    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--fractions", type=float, nargs="+",
                   default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


def main():
    args = parse_args()

    config = Config()
    config.seed = args.seed
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.lr = args.lr
    config.patience = args.patience

    run_sample_efficiency(args.data_path, args.fractions, config)


if __name__ == "__main__":
    main()
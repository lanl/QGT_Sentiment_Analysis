"""
Expressibility Analysis for QGT Quantum Circuits

This script computes the expressibility of parameterized quantum circuits (PQCs)
following the methodology from:
Sim, S., Johnson, P. D., & Aspuru-Guzik, A. (2019). 
"Expressibility and entangling capability of parameterized quantum circuits for hybrid quantum-classical algorithms."
Advanced Quantum Technologies, 2(12), 1900070.

Expressibility measures how well a PQC can explore the Hilbert space compared to 
Haar-random unitaries. Lower expressibility values indicate better coverage.
"""

import numpy as np
import pennylane as qml
from scipy.stats import entropy
from scipy.special import rel_entr
import matplotlib.pyplot as plt
from tqdm import tqdm

# ============================================================================
# Circuit Definitions
# ============================================================================

def create_qgt_circuit(n_qubits, include_qft=True, entanglement='ring'):
    """
    Create the QGT circuit for expressibility analysis.
    
    Args:
        n_qubits: Number of qubits
        include_qft: Whether to include QFT layer
        entanglement: 'ring' or 'linear'
    
    Returns:
        QNode function, number of parameters
    """
    dev = qml.device('default.qubit', wires=n_qubits)
    
    # Calculate number of parameters: 4 rotation layers * n_qubits
    n_params = 4 * n_qubits  # Rx, Ry, Rz, and final Ry
    
    @qml.qnode(dev)
    def circuit(params):
        """
        QGT-style PQC with:
        - 3 layers of Rx, Ry, Rz rotations
        - Ring/Linear CNOT entanglement
        - Additional Ry layer
        - Optional QFT
        """
        idx = 0
        
        # Layer 1: Rx rotations
        for i in range(n_qubits):
            qml.RX(params[idx + i], wires=i)
        idx += n_qubits
        
        # Layer 2: Ry rotations
        for i in range(n_qubits):
            qml.RY(params[idx + i], wires=i)
        idx += n_qubits
        
        # Layer 3: Rz rotations
        for i in range(n_qubits):
            qml.RZ(params[idx + i], wires=i)
        idx += n_qubits
        
        # Entanglement layer
        if entanglement == 'ring':
            # Ring-style CNOT
            for i in range(n_qubits):
                qml.CNOT(wires=[i, (i + 1) % n_qubits])
        elif entanglement == 'linear':
            # Linear CNOT
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
        
        # Additional Ry layer
        for i in range(n_qubits):
            qml.RY(params[idx + i], wires=i)
        
        # Optional QFT
        if include_qft:
            qml.QFT(wires=range(n_qubits))
        
        return qml.state()
    
    return circuit, n_params


def create_hardware_efficient_circuit(n_qubits, n_layers=2):
    """
    Standard hardware-efficient ansatz for comparison.
    """
    dev = qml.device('default.qubit', wires=n_qubits)
    
    @qml.qnode(dev)
    def circuit(params):
        idx = 0
        for layer in range(n_layers):
            # Rotation layer
            for i in range(n_qubits):
                qml.RY(params[idx], wires=i)
                qml.RZ(params[idx + 1], wires=i)
                idx += 2
            # Entanglement layer
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
        return qml.state()
    
    n_params = n_layers * n_qubits * 2
    return circuit, n_params


# ============================================================================
# Expressibility Computation
# ============================================================================

def compute_fidelity(state1, state2):
    """Compute fidelity between two pure states."""
    return np.abs(np.vdot(state1, state2)) ** 2


def sample_haar_fidelities(n_qubits, n_samples=5000):
    """
    Sample fidelities from Haar-random distribution.
    For pure states, this follows P(F) = (2^n - 1)(1 - F)^(2^n - 2)
    """
    dim = 2 ** n_qubits
    fidelities = []
    
    for _ in range(n_samples):
        # Generate two Haar-random states
        state1 = np.random.randn(dim) + 1j * np.random.randn(dim)
        state1 /= np.linalg.norm(state1)
        
        state2 = np.random.randn(dim) + 1j * np.random.randn(dim)
        state2 /= np.linalg.norm(state2)
        
        fidelities.append(compute_fidelity(state1, state2))
    
    return np.array(fidelities)


def sample_circuit_fidelities(circuit, n_params, n_samples=5000):
    """
    Sample fidelities from PQC by randomly sampling parameters.
    """
    fidelities = []
    
    for _ in tqdm(range(n_samples), desc="Sampling circuit fidelities"):
        # Random parameters uniformly in [0, 2π]
        params1 = np.random.uniform(0, 2 * np.pi, n_params)
        params2 = np.random.uniform(0, 2 * np.pi, n_params)
        
        state1 = circuit(params1)
        state2 = circuit(params2)
        
        fidelities.append(compute_fidelity(state1, state2))
    
    return np.array(fidelities)


def compute_expressibility(circuit_fidelities, haar_fidelities, n_bins=75):
    """
    Compute expressibility as KL divergence between circuit and Haar distributions.
    
    Lower values = more expressible (closer to Haar-random)
    """
    # Create histograms
    bins = np.linspace(0, 1, n_bins + 1)
    
    circuit_hist, _ = np.histogram(circuit_fidelities, bins=bins, density=True)
    haar_hist, _ = np.histogram(haar_fidelities, bins=bins, density=True)
    
    # Add small epsilon to avoid division by zero
    epsilon = 1e-10
    circuit_hist = circuit_hist + epsilon
    haar_hist = haar_hist + epsilon
    
    # Normalize
    circuit_hist = circuit_hist / circuit_hist.sum()
    haar_hist = haar_hist / haar_hist.sum()
    
    # KL divergence
    kl_div = entropy(circuit_hist, haar_hist)
    
    return kl_div, circuit_hist, haar_hist, bins


def theoretical_haar_distribution(fidelities, n_qubits):
    """
    Theoretical Haar distribution for pure states:
    P(F) = (d - 1)(1 - F)^(d - 2) where d = 2^n
    """
    d = 2 ** n_qubits
    return (d - 1) * (1 - fidelities) ** (d - 2)


# ============================================================================
# Trainability Analysis (Barren Plateau Detection)
# ============================================================================

def create_qgt_circuit_with_grad(n_qubits, include_qft=True, entanglement='ring'):
    """
    Create QGT circuit that returns expectation value for gradient computation.
    """
    dev = qml.device('default.qubit', wires=n_qubits)
    n_params = 4 * n_qubits
    
    @qml.qnode(dev, diff_method='parameter-shift')
    def circuit(params):
        idx = 0
        
        # Layer 1: Rx rotations
        for i in range(n_qubits):
            qml.RX(params[idx + i], wires=i)
        idx += n_qubits
        
        # Layer 2: Ry rotations
        for i in range(n_qubits):
            qml.RY(params[idx + i], wires=i)
        idx += n_qubits
        
        # Layer 3: Rz rotations
        for i in range(n_qubits):
            qml.RZ(params[idx + i], wires=i)
        idx += n_qubits
        
        # Entanglement layer
        if entanglement == 'ring':
            for i in range(n_qubits):
                qml.CNOT(wires=[i, (i + 1) % n_qubits])
        elif entanglement == 'linear':
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
        
        # Additional Ry layer
        for i in range(n_qubits):
            qml.RY(params[idx + i], wires=i)
        
        # Optional QFT
        if include_qft:
            qml.QFT(wires=range(n_qubits))
        
        # Return expectation value (cost function)
        return qml.expval(qml.PauliZ(0))
    
    return circuit, n_params


def compute_trainability(n_qubits, include_qft=True, entanglement='ring', n_samples=500):
    """
    Compute trainability by measuring gradient variance.
    
    Following McClean et al. (2018): "Barren plateaus in quantum neural network training landscapes"
    
    Trainability is assessed by:
    1. Sampling random parameter initializations
    2. Computing gradients at each point
    3. Measuring variance of gradients
    
    High variance = trainable (gradients provide useful signal)
    Low variance (exponentially small) = barren plateau (untrainable)
    
    Returns:
        mean_grad: Mean of gradient magnitudes
        var_grad: Variance of gradients (key metric)
        grad_samples: All gradient samples
    """
    circuit, n_params = create_qgt_circuit_with_grad(n_qubits, include_qft, entanglement)
    
    # Compute gradient function
    grad_fn = qml.grad(circuit)
    
    all_grads = []
    grad_norms = []
    
    for _ in tqdm(range(n_samples), desc=f"Computing gradients"):
        # Random parameters
        params = np.random.uniform(0, 2 * np.pi, n_params)
        
        # Compute gradient
        grads = grad_fn(params)
        
        all_grads.append(grads)
        grad_norms.append(np.linalg.norm(grads))
    
    all_grads = np.array(all_grads)
    
    # Compute statistics
    mean_grad = np.mean(np.abs(all_grads))
    var_grad = np.var(all_grads)  # Overall variance
    
    # Per-parameter variance (more detailed)
    per_param_var = np.var(all_grads, axis=0)
    mean_per_param_var = np.mean(per_param_var)
    
    # Gradient norm statistics
    mean_grad_norm = np.mean(grad_norms)
    std_grad_norm = np.std(grad_norms)
    
    return {
        'mean_grad': mean_grad,
        'var_grad': var_grad,
        'mean_per_param_var': mean_per_param_var,
        'per_param_var': per_param_var,
        'mean_grad_norm': mean_grad_norm,
        'std_grad_norm': std_grad_norm,
        'all_grads': all_grads
    }


def analyze_trainability_scaling(max_qubits=10, n_samples=300):
    """
    Analyze how gradient variance scales with number of qubits.
    Barren plateaus show exponential decay: Var(∂C/∂θ) ~ 2^(-n)
    """
    qubit_range = range(2, max_qubits + 1)
    
    results = {
        'qgt_full': [],
        'qgt_no_qft': [],
        'qgt_linear': [],
    }
    
    for n_q in qubit_range:
        print(f"\nAnalyzing {n_q} qubits...")
        
        # QGT Full
        res = compute_trainability(n_q, include_qft=True, entanglement='ring', n_samples=n_samples)
        results['qgt_full'].append(res['var_grad'])
        
        # QGT No QFT
        res = compute_trainability(n_q, include_qft=False, entanglement='ring', n_samples=n_samples)
        results['qgt_no_qft'].append(res['var_grad'])
        
        # QGT Linear
        res = compute_trainability(n_q, include_qft=True, entanglement='linear', n_samples=n_samples)
        results['qgt_linear'].append(res['var_grad'])
    
    return list(qubit_range), results


def plot_trainability_scaling(qubit_range, results, save_path=None):
    """
    Plot gradient variance vs number of qubits to detect barren plateaus.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.semilogy(qubit_range, results['qgt_full'], 'o-', label='QGT Full (Ring + QFT)', linewidth=2, markersize=8)
    ax.semilogy(qubit_range, results['qgt_no_qft'], 's-', label='QGT No QFT', linewidth=2, markersize=8)
    ax.semilogy(qubit_range, results['qgt_linear'], '^-', label='QGT Linear + QFT', linewidth=2, markersize=8)
    
    # Add theoretical barren plateau line (exponential decay)
    qubits = np.array(qubit_range)
    bp_line = results['qgt_full'][0] * (0.5 ** (qubits - qubits[0]))
    ax.semilogy(qubit_range, bp_line, 'k--', label='Exponential decay (barren plateau)', alpha=0.5)
    
    ax.set_xlabel('Number of Qubits', fontsize=12)
    ax.set_ylabel('Gradient Variance (log scale)', fontsize=12)
    ax.set_title('Trainability Analysis: Gradient Variance Scaling', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()
    return fig


# ============================================================================
# Entangling Capability
# ============================================================================

def meyer_wallach_entanglement(state, n_qubits):
    """
    Compute Meyer-Wallach entanglement measure Q.
    Q = 0 for product states, Q = 1 for maximally entangled.
    """
    dim = 2 ** n_qubits
    state = np.array(state).reshape(-1)
    
    Q = 0
    for k in range(n_qubits):
        # Compute reduced density matrix by tracing out qubit k
        # Reshape state into tensor
        state_tensor = state.reshape([2] * n_qubits)
        
        # Compute linear entropy for qubit k
        # Move qubit k to first position
        axes = list(range(n_qubits))
        axes.remove(k)
        axes = [k] + axes
        state_tensor = np.transpose(state_tensor, axes)
        
        # Reshape to separate qubit k from rest
        state_matrix = state_tensor.reshape(2, -1)
        
        # Reduced density matrix for qubit k
        rho_k = state_matrix @ state_matrix.conj().T
        
        # Linear entropy: S_L = 1 - Tr(rho^2)
        purity = np.real(np.trace(rho_k @ rho_k))
        linear_entropy = 1 - purity
        
        Q += linear_entropy
    
    # Normalize
    Q = (2 / n_qubits) * Q
    
    return Q


def compute_entangling_capability(circuit, n_params, n_qubits, n_samples=1000):
    """
    Compute average entangling capability of a circuit.
    """
    entanglements = []
    
    for _ in tqdm(range(n_samples), desc="Computing entangling capability"):
        params = np.random.uniform(0, 2 * np.pi, n_params)
        state = circuit(params)
        Q = meyer_wallach_entanglement(state, n_qubits)
        entanglements.append(Q)
    
    return np.mean(entanglements), np.std(entanglements), entanglements


# ============================================================================
# Main Analysis
# ============================================================================

def analyze_qgt_circuits(n_qubits=6, n_samples=2000):
    """
    Analyze expressibility and entangling capability of QGT circuit variants.
    """
    print(f"\n{'='*60}")
    print(f"QGT Circuit Expressibility Analysis ({n_qubits} qubits)")
    print(f"{'='*60}\n")
    
    results = {}
    
    # Circuit configurations to test
    configs = [
        ("QGT Full (Ring + QFT)", True, 'ring'),
        ("QGT No QFT (Ring)", False, 'ring'),
        ("QGT Linear Ent. + QFT", True, 'linear'),
        ("QGT Linear Ent., No QFT", False, 'linear'),
    ]
    
    # Sample Haar-random fidelities once
    print("Sampling Haar-random fidelities...")
    haar_fidelities = sample_haar_fidelities(n_qubits, n_samples)
    
    # Analyze each configuration
    for name, include_qft, entanglement in configs:
        print(f"\nAnalyzing: {name}")
        print("-" * 40)
        
        # Create circuit
        circuit, n_params = create_qgt_circuit(n_qubits, include_qft, entanglement)
        print(f"  Parameters: {n_params}")
        
        # Compute fidelities
        circuit_fidelities = sample_circuit_fidelities(circuit, n_params, n_samples)
        
        # Compute expressibility
        expr, circuit_hist, haar_hist, bins = compute_expressibility(
            circuit_fidelities, haar_fidelities
        )
        print(f"  Expressibility (KL div): {expr:.6f}")
        
        # Compute entangling capability
        ent_mean, ent_std, _ = compute_entangling_capability(
            circuit, n_params, n_qubits, n_samples=500
        )
        print(f"  Entangling Capability: {ent_mean:.4f} ± {ent_std:.4f}")
        
        results[name] = {
            'expressibility': expr,
            'entangling_mean': ent_mean,
            'entangling_std': ent_std,
            'n_params': n_params,
            'circuit_fidelities': circuit_fidelities,
            'circuit_hist': circuit_hist,
            'haar_hist': haar_hist,
            'bins': bins
        }
    
    # Add hardware-efficient baseline
    print(f"\nAnalyzing: Hardware-Efficient Ansatz (baseline)")
    print("-" * 40)
    circuit, n_params = create_hardware_efficient_circuit(n_qubits, n_layers=2)
    print(f"  Parameters: {n_params}")
    circuit_fidelities = sample_circuit_fidelities(circuit, n_params, n_samples)
    expr, circuit_hist, haar_hist, bins = compute_expressibility(
        circuit_fidelities, haar_fidelities
    )
    print(f"  Expressibility (KL div): {expr:.6f}")
    ent_mean, ent_std, _ = compute_entangling_capability(
        circuit, n_params, n_qubits, n_samples=500
    )
    print(f"  Entangling Capability: {ent_mean:.4f} ± {ent_std:.4f}")
    
    results["Hardware-Efficient (2L)"] = {
        'expressibility': expr,
        'entangling_mean': ent_mean,
        'entangling_std': ent_std,
        'n_params': n_params,
        'circuit_fidelities': circuit_fidelities,
        'circuit_hist': circuit_hist,
        'haar_hist': haar_hist,
        'bins': bins
    }
    
    return results, haar_fidelities


def plot_results(results, haar_fidelities, n_qubits, save_path=None):
    """
    Create visualization of expressibility analysis.
    """
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    
    # Plot fidelity distributions
    bins = np.linspace(0, 1, 76)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Theoretical Haar distribution
    haar_theoretical = theoretical_haar_distribution(bin_centers, n_qubits)
    haar_theoretical = haar_theoretical / haar_theoretical.sum() / (bins[1] - bins[0])
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    
    for idx, (name, data) in enumerate(results.items()):
        ax = axes[idx // 3, idx % 3]
        
        # Plot circuit histogram
        ax.hist(data['circuit_fidelities'], bins=bins, density=True, 
                alpha=0.7, label='Circuit', color=colors[idx])
        
        # Plot Haar reference
        ax.hist(haar_fidelities, bins=bins, density=True, 
                alpha=0.3, label='Haar', color='gray')
        
        # Plot theoretical Haar
        ax.plot(bin_centers, haar_theoretical, 'k--', label='Haar (theory)', linewidth=1.5)
        
        ax.set_xlabel('Fidelity')
        ax.set_ylabel('Probability Density')
        ax.set_title(f'{name}\nExpr={data["expressibility"]:.4f}, Ent={data["entangling_mean"]:.3f}')
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1)
    
    # Hide empty subplot if odd number of configs
    if len(results) < 6:
        axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")
    
    plt.show()
    
    return fig


def create_summary_table(results):
    """
    Create a summary table for the paper.
    """
    print("\n" + "=" * 70)
    print("SUMMARY TABLE (for paper)")
    print("=" * 70)
    print(f"{'Circuit Configuration':<30} {'Params':<8} {'Expr. (↓)':<12} {'Ent. Cap.':<15}")
    print("-" * 70)
    
    for name, data in results.items():
        print(f"{name:<30} {data['n_params']:<8} {data['expressibility']:<12.6f} "
              f"{data['entangling_mean']:.4f} ± {data['entangling_std']:.4f}")
    
    print("-" * 70)
    print("Expr. = Expressibility (KL divergence from Haar, lower is better)")
    print("Ent. Cap. = Meyer-Wallach entangling capability (0-1, higher = more entangled)")
    print("=" * 70)


def generate_latex_table(results):
    """
    Generate LaTeX table for the paper.
    """
    print("\n% LaTeX table for paper:")
    print(r"\begin{table}[t]")
    print(r"    \centering")
    print(r"    \caption{Expressibility and Entangling Capability Analysis}")
    print(r"    \label{tab:expressibility}")
    print(r"    \begin{tabular}{@{}lccc@{}}")
    print(r"    \toprule")
    print(r"    \textbf{Configuration} & \textbf{Params} & \textbf{Expressibility} $\downarrow$ & \textbf{Ent. Capability} \\")
    print(r"    \midrule")
    
    for name, data in results.items():
        # Escape underscores for LaTeX
        latex_name = name.replace("_", r"\_")
        print(f"    {latex_name} & {data['n_params']} & {data['expressibility']:.4f} & "
              f"${data['entangling_mean']:.3f} \\pm {data['entangling_std']:.3f}$ \\\\")
    
    print(r"    \bottomrule")
    print(r"    \end{tabular}")
    print(r"\end{table}")


# ============================================================================
# Run Analysis
# ============================================================================

if __name__ == "__main__":
    # Use 6 qubits for faster computation (same as GloVe experiments)
    # For full BERT (10 qubits), increase but expect longer runtime
    N_QUBITS = 6
    N_SAMPLES = 2000  # Increase for more accurate results
    
    print("=" * 60)
    print("PQC Expressibility Analysis for QGT Paper")
    print("Based on Sim et al. (2019) methodology")
    print("=" * 60)
    
    # Run expressibility analysis
    results, haar_fidelities = analyze_qgt_circuits(
        n_qubits=N_QUBITS, 
        n_samples=N_SAMPLES
    )
    
    # Print summary
    create_summary_table(results)
    
    # Generate LaTeX table
    generate_latex_table(results)
    
    # Plot results
    plot_results(
        results, 
        haar_fidelities, 
        N_QUBITS,
        save_path='expressibility_analysis.png'
    )
    
    
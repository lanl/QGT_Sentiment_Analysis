# QGT-NLP
Hybrid Quantum–Classical Graph Transformers for Efficient Sentiment Analysis

## Overview

This repository contains the implementation of the **Quantum Graph Transformer (QGT)**, a hybrid quantum–classical model for sentiment analysis. The QGT replaces classical linear attention projections with parameterized quantum circuits (PQCs), achieving comparable accuracy to its classical counterpart (CGT) while using **29.4× fewer parameters** — 1,620 vs 47,620. The quantum attention mechanism uses only **80 trainable parameters** compared to 46,080 in the classical version.

| Model | Accuracy | Total Params | Attention Params |
|-------|----------|-------------|-----------------|
| **QGT** | **91%** | 1,620 | 80 |
| CGT | 88% | 47,620 | 46,080 |

The QGT encodes sentences as graphs where each word is a node with pre-trained embeddings (BERT or GloVe). The quantum attention mechanism uses amplitude encoding into 10-qubit states, a parameterized quantum circuit (RX/RY/RZ rotations, ring CNOT entanglement, QFT), and Pauli X/Y/Z measurements to produce query and key vectors for scaled dot-product graph attention.

### Python scripts

- `train_qgt_cgt.py` is the main training script. It trains both the QGT and CGT models on preprocessed graph data. The QGT uses parameterized quantum circuits for Q/K generation; the CGT uses classical linear projections. Ablation flags allow disabling individual quantum components (QFT, entanglement, RL regularization, LayerNorm). Both models use the same graph structure, embeddings, message passing, and training procedure — the only difference is the attention mechanism.

- `preprocess.py` converts raw tab-separated text files into graph-structured PyTorch Geometric `Data` objects. Supports BERT (768-dim, 10 qubits), GloVe (50-dim, 6 qubits), and Word2Vec (300-dim, 9 qubits) embeddings, and several graph construction strategies (fully connected, chain, k-nearest neighbor). Outputs are saved as pickle files.

- `expressibility_analysis.py` computes the expressibility, trainability, and entangling capability of the QGT's parameterized quantum circuits following the methodology from Sim et al. (2019) and McClean et al. (2018). Expressibility is measured as KL divergence from the Haar-random distribution. Trainability is assessed via gradient variance scaling to detect barren plateaus. Entangling capability uses the Meyer-Wallach measure.

- `train_qae.py` implements a Quantum Autoencoder (QAE) that compresses the 10-qubit representation down to 6 qubits, followed by training a 6-qubit QGT. The QAE is pre-trained to minimize trash qubit entropy, then the compressed latent representation is used for Q/K generation via angle embedding.

- `sample_efficiency_qgt.py` and `sample_efficiency_CGT_Linear.py` run sample efficiency experiments, training the QGT and CGT (linear Q/K) respectively across varying fractions of the training data (10% to 100%).

- `Analysis.ipynb` contains results visualization and comparison plots.

### Directories with data

- `data/` contains the sentiment analysis datasets used in the experiments. These are tab-separated text files with format `text\tlabel` (binary sentiment, 0 = negative, 1 = positive).

| Dataset | File | Samples |
|---------|------|---------|
| Yelp | `yelp_labelled.txt` | 1,000 |
| IMDB | `imdb_labelled.txt` | 1,000 |
| Amazon | `amazon_cells_labelled.txt` | 1,000 |
| Meaning Classification | `mc_full.txt` | 100 |
| Relative Pronoun | `rp_full.txt` | 104 |

## Installation

```bash
git clone https://github.com/[YOUR-USERNAME]/QGT-NLP.git
cd QGT-NLP
pip install -r requirements.txt
```

## Usage

### Preprocessing

```bash
# BERT embeddings, fully connected graph
python preprocess.py --data_path original/yelp_labelled.txt --embedding bert --graph full

# GloVe embeddings, KNN graph
python preprocess.py --data_path original/yelp_labelled.txt --embedding glove --graph knn-5

# Multiple graph types
python preprocess.py --data_path original/yelp_labelled.txt --embedding bert --graph full knn-5 chain

# Quick test with small subset
python preprocess.py --data_path original/yelp_labelled.txt --embedding bert --graph full --small 100
```

Output is saved to `preprocessed_data/{dataset}/{embedding}/{graph}/data.pkl`.

### Training QGT and CGT

```bash
# Train both QGT and CGT
python train_qgt_cgt.py --data_path preprocessed_data/yelp_labelled/bert/full/data.pkl

# QGT only
python train_qgt_cgt.py --data_path preprocessed_data/yelp_labelled/bert/full/data.pkl --qgt_only

# CGT only
python train_qgt_cgt.py --data_path preprocessed_data/yelp_labelled/bert/full/data.pkl --cgt_only
```

Ablation flags for QGT:
```bash
--no_qft                 # Remove QFT layer
--no_entanglement        # Remove CNOT gates entirely
--linear_entanglement    # Linear CNOT chain instead of ring
--no_rl_reg              # Disable RL regularization
--no_layernorm           # Disable LayerNorm after attention
```

### PQC expressibility analysis

```bash
python expressibility_analysis.py
```

### Quantum Autoencoder compression (10 → 6 qubits)

```bash
# Full pipeline: QAE pre-training then 6-qubit QGT
python train_qae.py --data_path preprocessed_data/yelp_labelled/bert/full/data.pkl --qae_epochs 50 --epochs 50

# Load pre-trained QAE encoder
python train_qae.py --data_path data.pkl --skip_qae --qae_checkpoint saved_models_qgt_6qubit/qae_encoder.pt
```

### Sample efficiency analysis

```bash
# QGT across training fractions
python sample_efficiency_qgt.py --data_path preprocessed_data/yelp_labelled/bert/full/data.pkl

# CGT (linear Q/K) across training fractions
python sample_efficiency_CGT_Linear.py --data_path preprocessed_data/yelp_labelled/bert/full/data.pkl
```

## requirements.txt and used Python libraries

```
torch>=2.0.0
torch-geometric>=2.4.0
torch-scatter>=2.1.0
pennylane>=0.35.0
pennylane-lightning>=0.35.0
transformers>=4.30.0
numpy>=1.24.0
scipy>=1.10.0
matplotlib>=3.7.0
tqdm>=4.65.0
nltk>=3.8.0
gensim>=4.3.0
scikit-learn>=1.2.0
```

## Copyright Notice:
© 2025. Triad National Security, LLC. All rights reserved.
This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos
National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S.
Department of Energy/National Nuclear Security Administration. All rights in the program are
reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear
Security Administration. The Government is granted for itself and others acting on its behalf a
nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare
derivative works, distribute copies to the public, perform publicly and display publicly, and to permit
others to do so.

**LANL C Number: C22038**

## License:
This program is open source under the BSD-3 License.
Redistribution and use in source and binary forms, with or without modification, are permitted
provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and
the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions
and the following disclaimer in the documentation and/or other materials provided with the
distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse
or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
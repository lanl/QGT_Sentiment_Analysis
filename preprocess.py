#!/usr/bin/env python3
"""
QGT Preprocessing Script
=========================
Preprocesses text data with different embeddings and graph structures.

Embeddings:
- BERT (768-dim) → 10 qubits, amp_dim=1024
- GloVe (50-dim) → 6 qubits, amp_dim=64
- Word2Vec (300-dim) → 9 qubits, amp_dim=512

Graph Structures:
- full: Fully connected graph
- chain: Linear chain (sequential connections)
- knn-3, knn-4, knn-5: k-nearest neighbor graphs

Output: Saves preprocessed PyG Data objects to disk for consistent training.
"""

import os
import sys
import json
import pickle
import argparse
import random
import numpy as np
import torch
from datetime import datetime
from tqdm import tqdm

import nltk
from nltk.tokenize import word_tokenize

from torch_geometric.data import Data

# =============================================================================
# CONFIGURATION
# =============================================================================
DEFAULT_CONFIG = {
    "seed": 42,
    "max_len": 45,
    "bert_model": "bert-base-uncased",
    "glove_path": "glove.6B.50d.txt",
    "w2v_model": "word2vec-google-news-300",
    "output_dir": "preprocessed_data",
}

EMBEDDING_CONFIGS = {
    "bert": {"dim": 768, "n_qubits": 10, "amp_dim": 1024},
    "glove": {"dim": 50, "n_qubits": 6, "amp_dim": 64},
    "word2vec": {"dim": 300, "n_qubits": 9, "amp_dim": 512},
}

GRAPH_TYPES = ["full", "chain", "knn-3", "knn-4", "knn-5"]

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# NLTK setup
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)

# =============================================================================
# GRAPH CONSTRUCTION
# =============================================================================
def build_full_edges(max_len, n_real):
    """Fully connected graph for real tokens."""
    src, dst = [], []
    for i in range(n_real):
        for j in range(n_real):
            if i != j:
                src.append(j)
                dst.append(i)
    # Self-loops for all nodes
    for i in range(max_len):
        src.append(i)
        dst.append(i)
    return torch.tensor([src, dst], dtype=torch.long)


def build_chain_edges(max_len, n_real):
    """Linear chain graph (sequential connections)."""
    src, dst = [], []
    for i in range(n_real - 1):
        # Bidirectional edges
        src.extend([i, i + 1])
        dst.extend([i + 1, i])
    # Self-loops for all nodes
    for i in range(max_len):
        src.append(i)
        dst.append(i)
    return torch.tensor([src, dst], dtype=torch.long)


def build_knn_edges(x_real, max_len, k):
    """k-nearest neighbor graph based on embedding similarity."""
    n_real = x_real.size(0)
    src, dst = [], []
    
    if n_real <= 1:
        for i in range(max_len):
            src.append(i)
            dst.append(i)
        return torch.tensor([src, dst], dtype=torch.long)
    
    # Compute cosine similarity
    x_norm = x_real / x_real.norm(dim=1, keepdim=True).clamp_min(1e-8)
    sim = x_norm @ x_norm.t()
    
    for i in range(n_real):
        sim_i = sim[i].clone()
        sim_i[i] = -1e9  # Exclude self
        kk = min(k, n_real - 1)
        neighbors = torch.topk(sim_i, kk).indices.tolist()
        for j in neighbors:
            # Bidirectional edges
            src.extend([j, i])
            dst.extend([i, j])
    
    # Remove duplicates
    edge_set = set(zip(src, dst))
    src = [e[0] for e in edge_set]
    dst = [e[1] for e in edge_set]
    
    # Self-loops for all nodes
    for i in range(max_len):
        src.append(i)
        dst.append(i)
    
    return torch.tensor([src, dst], dtype=torch.long)


def build_edges(x_real, max_len, graph_type):
    """Build edges based on graph type."""
    n_real = x_real.size(0) if isinstance(x_real, torch.Tensor) else x_real
    
    if graph_type == "full":
        return build_full_edges(max_len, n_real)
    elif graph_type == "chain":
        return build_chain_edges(max_len, n_real)
    elif graph_type.startswith("knn-"):
        k = int(graph_type.split("-")[1])
        return build_knn_edges(x_real, max_len, k)
    else:
        raise ValueError(f"Unknown graph type: {graph_type}")

# =============================================================================
# EMBEDDING LOADERS
# =============================================================================
class BERTEmbedder:
    """BERT embedding extractor."""
    
    def __init__(self, model_name="bert-base-uncased", device="cpu"):
        from transformers import BertTokenizer, BertModel
        
        print(f"Loading BERT model: {model_name}...")
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        self.device = device
        for p in self.model.parameters():
            p.requires_grad = False
    
    @torch.no_grad()
    def get_embeddings(self, text, max_len=45):
        encoded = self.tokenizer(
            text, return_tensors="pt", max_length=max_len,
            padding="max_length", truncation=True, return_attention_mask=True
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        
        outputs = self.model(input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state.squeeze(0)  # (seq_len, 768)
        pad_mask = attention_mask.squeeze(0).float()
        
        return embeddings.cpu(), pad_mask.cpu()


class GloVeEmbedder:
    """GloVe embedding extractor."""
    
    def __init__(self, filepath, dim=50):
        print(f"Loading GloVe from {filepath}...")
        self.embeddings = {}
        self.dim = dim
        
        with open(filepath, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc="Loading GloVe"):
                parts = line.strip().split()
                if len(parts) != dim + 1:
                    continue
                word = parts[0]
                vec = np.array([float(x) for x in parts[1:]], dtype=np.float32)
                self.embeddings[word] = vec
        
        # Compute mean for OOV
        all_vecs = list(self.embeddings.values())
        self.mean_vec = np.mean(all_vecs, axis=0)
        print(f"Loaded {len(self.embeddings)} GloVe vectors")
    
    def get_embeddings(self, text, max_len=45):
        tokens = word_tokenize(text.lower())[:max_len]
        n_real = len(tokens)
        
        embeddings = []
        for token in tokens:
            if token in self.embeddings:
                embeddings.append(self.embeddings[token])
            else:
                embeddings.append(self.mean_vec)
        
        # Pad to max_len
        while len(embeddings) < max_len:
            embeddings.append(np.zeros(self.dim, dtype=np.float32))
        
        x = torch.tensor(np.array(embeddings), dtype=torch.float32)
        pad_mask = torch.zeros(max_len)
        pad_mask[:n_real] = 1.0
        
        return x, pad_mask


class Word2VecEmbedder:
    """Word2Vec embedding extractor."""
    
    def __init__(self, model_name="word2vec-google-news-300"):
        import gensim.downloader as api
        
        print(f"Loading Word2Vec model: {model_name}...")
        print("(This may take a few minutes on first run)")
        self.model = api.load(model_name)
        self.dim = self.model.vector_size
        
        # Compute mean for OOV
        sample_words = list(self.model.key_to_index.keys())[:10000]
        self.mean_vec = np.mean([self.model[w] for w in sample_words], axis=0)
        print(f"Loaded Word2Vec with {len(self.model.key_to_index)} words, {self.dim}d")
    
    def get_embeddings(self, text, max_len=45):
        tokens = word_tokenize(text.lower())[:max_len]
        n_real = len(tokens)
        
        embeddings = []
        for token in tokens:
            if token in self.model:
                embeddings.append(self.model[token])
            else:
                embeddings.append(self.mean_vec)
        
        # Pad to max_len
        while len(embeddings) < max_len:
            embeddings.append(np.zeros(self.dim, dtype=np.float32))
        
        x = torch.tensor(np.array(embeddings), dtype=torch.float32)
        pad_mask = torch.zeros(max_len)
        pad_mask[:n_real] = 1.0
        
        return x, pad_mask

# =============================================================================
# DATA LOADING
# =============================================================================
def load_raw_data(filepath):
    """Load raw text data from file."""
    texts, labels = [], []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "\t" not in line:
                continue
            text, label = line.split("\t")
            texts.append(text)
            labels.append(int(label))
    return texts, labels


def stratified_split(labels, train_frac=0.7, val_frac=0.1, seed=42):
    """Split data with stratification."""
    rng = np.random.default_rng(seed)
    labels = np.array(labels)
    
    idx0 = np.where(labels == 0)[0]
    idx1 = np.where(labels == 1)[0]
    rng.shuffle(idx0)
    rng.shuffle(idx1)
    
    def split(idxs, tr, va):
        n = len(idxs)
        n_tr, n_va = int(n * tr), int(n * va)
        return idxs[:n_tr], idxs[n_tr:n_tr+n_va], idxs[n_tr+n_va:]
    
    tr0, va0, te0 = split(idx0, train_frac, val_frac)
    tr1, va1, te1 = split(idx1, train_frac, val_frac)
    
    train = np.concatenate([tr0, tr1])
    val = np.concatenate([va0, va1])
    test = np.concatenate([te0, te1])
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    
    return train.tolist(), val.tolist(), test.tolist()

# =============================================================================
# PREPROCESSING
# =============================================================================
def preprocess_dataset(
    data_path,
    embedding_type,
    graph_types,
    output_dir,
    config,
    small_sample=None
):
    """
    Preprocess dataset with specified embedding and graph types.
    
    Args:
        data_path: Path to raw data file
        embedding_type: 'bert', 'glove', or 'word2vec'
        graph_types: List of graph types to generate
        output_dir: Output directory
        config: Configuration dict
        small_sample: If int, use only this many samples
    """
    set_seed(config["seed"])
    
    # Load raw data
    print(f"\nLoading data from {data_path}...")
    texts, labels = load_raw_data(data_path)
    print(f"Loaded {len(texts)} samples")
    
    # Small sample option
    if small_sample and small_sample < len(texts):
        print(f"Using small sample: {small_sample} samples")
        indices = list(range(len(texts)))
        random.shuffle(indices)
        indices = indices[:small_sample]
        texts = [texts[i] for i in indices]
        labels = [labels[i] for i in indices]
    
    # Get embedding config
    emb_config = EMBEDDING_CONFIGS[embedding_type]
    
    # Initialize embedder
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if embedding_type == "bert":
        embedder = BERTEmbedder(config["bert_model"], device)
    elif embedding_type == "glove":
        embedder = GloVeEmbedder(config["glove_path"], emb_config["dim"])
    elif embedding_type == "word2vec":
        embedder = Word2VecEmbedder(config["w2v_model"])
    else:
        raise ValueError(f"Unknown embedding type: {embedding_type}")
    
    # Extract embeddings
    print(f"\nExtracting {embedding_type} embeddings...")
    all_embeddings = []
    all_pad_masks = []
    
    for text in tqdm(texts, desc="Extracting embeddings"):
        x, pad_mask = embedder.get_embeddings(text, config["max_len"])
        all_embeddings.append(x)
        all_pad_masks.append(pad_mask)
    
    # Create split indices
    train_idx, val_idx, test_idx = stratified_split(labels, seed=config["seed"])
    
    # Create output directory
    dataset_name = os.path.splitext(os.path.basename(data_path))[0]
    sample_suffix = f"_small{small_sample}" if small_sample else ""
    base_output_dir = os.path.join(output_dir, f"{dataset_name}{sample_suffix}", embedding_type)
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Process each graph type
    for graph_type in graph_types:
        print(f"\nBuilding {graph_type} graphs...")
        graph_output_dir = os.path.join(base_output_dir, graph_type)
        os.makedirs(graph_output_dir, exist_ok=True)
        
        # Build Data objects
        data_list = []
        for i, (x, pad_mask, label) in enumerate(tqdm(
            zip(all_embeddings, all_pad_masks, labels),
            total=len(texts),
            desc=f"Building {graph_type} graphs"
        )):
            n_real = int(pad_mask.sum().item())
            
            # Build edges (need real embeddings for knn)
            if graph_type.startswith("knn"):
                edge_index = build_edges(x[:n_real], config["max_len"], graph_type)
            else:
                edge_index = build_edges(n_real, config["max_len"], graph_type)
            
            data = Data(
                x=x,
                edge_index=edge_index,
                y=torch.tensor(label, dtype=torch.long),
                pad_mask=pad_mask,
                idx=i
            )
            data_list.append(data)
        
        # Save data
        save_dict = {
            "data_list": data_list,
            "train_idx": train_idx,
            "val_idx": val_idx,
            "test_idx": test_idx,
            "config": {
                **config,
                "embedding_type": embedding_type,
                "graph_type": graph_type,
                "emb_dim": emb_config["dim"],
                "n_qubits": emb_config["n_qubits"],
                "amp_dim": emb_config["amp_dim"],
                "n_samples": len(texts),
                "n_train": len(train_idx),
                "n_val": len(val_idx),
                "n_test": len(test_idx),
            }
        }
        
        save_path = os.path.join(graph_output_dir, "data.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(save_dict, f)
        
        print(f"Saved to {save_path}")
    
    # Save metadata
    metadata = {
        "dataset": dataset_name,
        "embedding_type": embedding_type,
        "graph_types": graph_types,
        "n_samples": len(texts),
        "small_sample": small_sample,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "timestamp": datetime.now().isoformat(),
    }
    
    meta_path = os.path.join(base_output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nPreprocessing complete!")
    print(f"Output directory: {base_output_dir}")
    print(f"Samples: {len(texts)} (Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)})")

# =============================================================================
# MAIN
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="QGT Preprocessing Script")
    
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to raw data file (e.g., yelp_labelled.txt)")
    parser.add_argument("--embedding", type=str, choices=["bert", "glove", "word2vec", "all"],
                        default="all", help="Embedding type to use")
    parser.add_argument("--graph", type=str, nargs="+", 
                        choices=GRAPH_TYPES + ["all"], default=["all"],
                        help="Graph types to generate")
    parser.add_argument("--output_dir", type=str, default="preprocessed_data",
                        help="Output directory")
    parser.add_argument("--small", type=int, default=None,
                        help="Use small sample (e.g., 100 for 100 samples)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max_len", type=int, default=45, help="Max sequence length")
    parser.add_argument("--glove_path", type=str, default="glove.6B.50d.txt",
                        help="Path to GloVe file")
    
    return parser.parse_args()


def main():
    # Check if running in notebook
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            print("📓 Running in Jupyter notebook - using default config")
            # Default config for notebook
            config = DEFAULT_CONFIG.copy()
            
            # Example usage in notebook:
            # preprocess_dataset("original/yelp_labelled.txt", "bert", ["full", "knn-5"], "preprocessed_data", config)
            print("\nExample usage:")
            print('  preprocess_dataset("original/yelp_labelled.txt", "bert", ["full"], "preprocessed_data", config)')
            print('  preprocess_dataset("original/yelp_labelled.txt", "bert", ["full"], "preprocessed_data", config, small_sample=100)')
            return
    except:
        pass
    
    args = parse_args()
    
    # Build config
    config = DEFAULT_CONFIG.copy()
    config["seed"] = args.seed
    config["max_len"] = args.max_len
    config["glove_path"] = args.glove_path
    
    # Determine embeddings to process
    if args.embedding == "all":
        embeddings = ["bert", "glove", "word2vec"]
    else:
        embeddings = [args.embedding]
    
    # Determine graph types
    if "all" in args.graph:
        graph_types = GRAPH_TYPES
    else:
        graph_types = args.graph
    
    # Process each embedding type
    for emb_type in embeddings:
        print(f"\n{'='*60}")
        print(f"Processing {emb_type.upper()} embeddings")
        print(f"{'='*60}")
        
        preprocess_dataset(
            data_path=args.data_path,
            embedding_type=emb_type,
            graph_types=graph_types,
            output_dir=args.output_dir,
            config=config,
            small_sample=args.small
        )


if __name__ == "__main__":
    main()

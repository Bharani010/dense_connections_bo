# ~/MerCBO/MerCBO/experiments/test_functions/transformer_perplexity.py
import sys
import pathlib
import torch
import numpy as np 
from typing import List

TRANS_ROOT = pathlib.Path("/home/sveerepa/MerCBO/pruned_dense_transformer_gpu1/")
sys.path.append(str(TRANS_ROOT))

from config import get_config
from train import train_model 
from grid_search import dense_connections

class TransformerPerplexity:
    def __init__(self):
        self.patterns: List[str] = list(dense_connections.keys())
        k: int = len(self.patterns)  # k = 10

        # --- Define as 10 independent binary variables ---
        self.n_vertices = [2] * k  # Python list of 10 twos, e.g., [2, 2, ..., 2]

        # --- Adjacency matrices: one 2x2 for each binary variable ---
        adj2 = torch.tensor([[0., 1.], [1., 0.]], dtype=torch.float32)
        self.adjacency_mat = [adj2.clone() for _ in range(k)] # List of 10 (2x2) tensors

        # --- Fourier features: one set for each binary variable ---
        self.fourier_freq  = [torch.ones(2, dtype=torch.float32) for _ in range(k)] # List of 10 (2,) tensors
        self.fourier_basis = [torch.eye(2, dtype=torch.float32)  for _ in range(k)] # List of 10 (2x2) tensors

        # --- Base Transformer Config ---
        base = get_config()
        base["num_epochs"] = 15      # For smoke test
        base["preload"]    = None
        base["d_models"]   = [512]  # Fixed for this optimization
        base["num_heads"]  = [2]    # Fixed for this optimization

        if 'lang_src' not in base or base['lang_src'] != 'en':
            base['lang_src'] = 'en'
        if 'lang_tgt' not in base or base['lang_tgt'] != 'it':
            base['lang_tgt'] = 'it'
        self.base_cfg      = base

        # --- Warm-start: One-hot vectors for pattern 0 and pattern 1 ---
        # suggested_init shape should be (num_initial_points, num_binary_variables) -> (2, 10)
        eye_k = torch.eye(k, dtype=torch.long) # Use k for clarity
        self.suggested_init = torch.vstack([eye_k[0], eye_k[1]])

    def evaluate(self, x_vec: torch.Tensor) -> torch.Tensor:
        """
        x_vec: A LongTensor of shape (10,) with 0s and one 1 (ideally),
               or a vector of 0s/1s if AFO proposes non-one-hot.
               Represents the chosen skip pattern.
        """
        # Ensure input is a 1D tensor of length k
        if not (x_vec.dim() == 1 and x_vec.size(0) == len(self.patterns)):
             raise ValueError(f"evaluate() expects a 1D tensor of length {len(self.patterns)}, got {x_vec.shape}")

        # Handle cases where MerCBO's AFO might not return a strict one-hot vector
        # when optimizing multiple binary variables. Project to the most likely one-hot.
        if x_vec.sum().item() != 1:
            print(f"Warning: MerCBO proposed x_vec {x_vec} with sum {x_vec.sum().item()}. Projecting to one-hot via argmax.")
            idx = torch.argmax(x_vec).item() # Choose the index with the highest value (or first if ties)
        else: # It's one-hot
            idx = int(torch.argmax(x_vec).item()) # Find the index of the '1'

        pattern_key = self.patterns[idx]

        cfg = self.base_cfg.copy()
        cfg["dense_connections"] = dense_connections[pattern_key]
        
        d_model_val = cfg['d_models'][0] 
        num_heads_val = cfg['num_heads'][0] if isinstance(cfg['num_heads'], list) else cfg['num_heads']


        cfg["model_folder"] = f"mercbo_{pattern_key}_emb{d_model_val}_head{num_heads_val}"
        cfg["experiment_name"] = f"runs/{cfg['model_folder']}"

        print(f"\n? Evaluating Transformer: Pattern={pattern_key} (Index {idx}), d_model={d_model_val}, Heads={num_heads_val}")
        
        perplexity = train_model(cfg)
        
        # MerCBO maximizes; perplexity should be minimized. Return negative perplexity.
        return torch.tensor([-float(perplexity)], dtype=torch.float32)
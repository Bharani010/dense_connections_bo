# File: /home/sveerepa/MerCBO/MerCBO/experiments/test_functions/EncoderBO.py

import sys
import pathlib
import torch
import numpy as np
from typing import List

# --- Point Python to your transformer project so we can import its modules ---
TRANS_ROOT = pathlib.Path("/home/sveerepa/pruned_dense_transformer_gpu1/Code/")
sys.path.append(str(TRANS_ROOT))

# Now we can import your custom functions
from config import get_config
from train import train_model
from nas_utils import is_connected # The connectivity check function we created

class EncoderBO:
    """
    This class is the main bridge between the MerCBO framework and our Transformer.
    It defines the entire search problem: what are the choices (the search space),
    and how do we get a score for any given choice (the evaluate method).
    """
    def __init__(self):
        # --- 1. Base Transformer Configuration ---
        self.base_cfg = get_config()
        self.base_cfg["num_epochs"] = 15
        self.base_cfg["preload"] = None

        # --- 2. Defining the Full Search Space for Bayesian Optimization ---
        N = self.base_cfg['num_layers']
        self.num_connections = (N + 1) * N // 2
        self.d_model_options = [64, 128, 256, 512]
        self.head_options = [2, 4, 8, 16]
        
        self.n_vertices = (
            ([2] * self.num_connections) +
            [len(self.d_model_options)] +
            [len(self.head_options)]
        )
        
        # --- 3. Graph Structures for MerCBO's Kernel ---
        self.adjacency_mat, self.fourier_freq, self.fourier_basis = self._build_graph_structures()

        # --- 4. The Automated 16-Point Warm-Up ---
        self.suggested_init = self._generate_warmup_points()
        print(f"Generated {self.suggested_init.shape[0]} warm-up points for the BO.")

    def _build_graph_structures(self):
        """A helper function to create the graph definition lists for MerCBO's kernel."""
        adj_mats, freqs, bases = [], [], []
        
        adj2 = torch.tensor([[0., 1.], [1., 0.]], dtype=torch.float32)
        freq2 = torch.ones(2, dtype=torch.float32)
        basis2 = torch.eye(2, dtype=torch.float32)
        for _ in range(self.num_connections):
            adj_mats.append(adj2.clone())
            freqs.append(freq2.clone())
            bases.append(basis2.clone())
            
        num_d_model_options = len(self.d_model_options)
        adj_d_model = torch.ones(num_d_model_options, num_d_model_options); adj_d_model.fill_diagonal_(0)
        adj_mats.append(adj_d_model)
        freqs.append(torch.arange(num_d_model_options, dtype=torch.float32) + 1)
        bases.append(torch.eye(num_d_model_options, dtype=torch.float32))
        
        num_head_options = len(self.head_options)
        adj_heads = torch.ones(num_head_options, num_head_options); adj_heads.fill_diagonal_(0)
        adj_mats.append(adj_heads)
        freqs.append(torch.arange(num_head_options, dtype=torch.float32) + 1)
        bases.append(torch.eye(num_head_options, dtype=torch.float32))
        
        return adj_mats, freqs, bases

    def _generate_warmup_points(self) -> torch.Tensor:
        """This function automatically generates the valid warm-up configurations."""
        N = self.base_cfg['num_layers']
        initial_points = []
        seq_matrix = np.zeros((N + 1, N + 1), dtype=int)
        for i in range(N):
            seq_matrix[i, i + 1] = 1
        seq_vector = self._matrix_to_vector(seq_matrix)

        for d_model_idx, d_model in enumerate(self.d_model_options):
            for head_idx, num_heads in enumerate(self.head_options):
                if d_model % num_heads == 0:
                    init_vector = np.concatenate([seq_vector, [d_model_idx, head_idx]])
                    initial_points.append(torch.tensor(init_vector, dtype=torch.long))
        
        return torch.stack(initial_points)

    def _vector_to_matrix(self, vector: np.ndarray) -> np.ndarray:
        """Helper to convert a flat binary vector back into a connection matrix."""
        N = self.base_cfg['num_layers']
        matrix = np.zeros((N + 1, N + 1), dtype=int)
        iu = np.triu_indices(N + 1, k=1)
        matrix[iu] = vector
        return matrix

    def _matrix_to_vector(self, matrix: np.ndarray) -> np.ndarray:
        """Helper to flatten a connection matrix into a binary vector."""
        N = self.base_cfg['num_layers']
        iu = np.triu_indices(N + 1, k=1)
        return matrix[iu]

    # This method must be indented correctly to belong to the EncoderBO class
    def evaluate(self, x_vec: torch.Tensor) -> torch.Tensor:
        """
        This is the main function called by MerCBO for each trial.
        It decodes the full vector, validates the architecture, trains the model, and returns perplexity.
        """
        # 1. Decode the full vector from BO into its components.
        connections_vector = x_vec[:self.num_connections].numpy().astype(int)
        d_model_idx = int(x_vec[self.num_connections].item())
        head_idx = int(x_vec[self.num_connections + 1].item())
        
        d_model_val = self.d_model_options[d_model_idx]
        head_val = self.head_options[head_idx]

        # 2. Validate the proposed architecture.
        if d_model_val % head_val != 0:
            print(f"Skipping invalid architecture: d_model={d_model_val} not divisible by num_heads={head_val}")
            return torch.tensor([-1e9], dtype=torch.float32)

        connections_matrix = self._vector_to_matrix(connections_vector)

        if not is_connected(connections_matrix):
            print(f"Skipping disconnected architecture: {connections_vector}")
            return torch.tensor([-1e9], dtype=torch.float32)

        # 3. If the architecture is valid, configure and train the model.
        cfg = self.base_cfg.copy()
        cfg["connections_matrix"] = connections_matrix
        cfg["d_models"] = [d_model_val]
        cfg["num_heads"] = head_val
        
        arch_hash = hash(tuple(x_vec.numpy()))
        cfg["model_folder"] = f"nas_run_{arch_hash}"
        cfg["experiment_name"] = f"runs/nas/{arch_hash}"
        
        with open("architecture_map.log", "a") as f:
            f.write(f"{arch_hash},{','.join(map(str, x_vec.numpy().tolist()))}\n")

        print(f"\n? Evaluating Architecture: d_model={d_model_val}, heads={head_val}, connections={connections_vector}")
        
        perplexity = train_model(cfg)

        # 4. Return the score to MerCBO (negative perplexity for maximization).
        return torch.tensor([-float(perplexity)], dtype=torch.float32)
# File: /home/sveerepa/pruned_dense_transformer_gpu1/Code/nas_utils.py

import numpy as np
from collections import deque

def is_connected(adjacency_matrix: np.ndarray) -> bool:
    """
    Checks if a path exists from the input (node 0) to the output of the last layer
    in a Directed Acyclic Graph (DAG) represented by an adjacency matrix.

    This function uses a Breadth-First Search (BFS) algorithm.

    Args:
        adjacency_matrix: An (N+1)x(N+1) upper-triangular binary matrix representing the DAG
                          where N is the number of layers.
                          A[i, j] = 1 means a connection from the output of node i to the input of layer j.
                          Node 0 represents the initial encoder input (after embeddings).
                          Nodes 1 to N represent the outputs of layers 1 to N.

    Returns:
        True if the output of the last layer (node N) is reachable from the input (node 0).
        False otherwise, indicating a disconnected architecture.
    """
    # Number of nodes in the graph is N_layers + 1 (for the input node)
    num_nodes = adjacency_matrix.shape[0]
    
    # The input node is 0. The target node is the output of the last layer, which is node N.
    # For a matrix of shape (N+1, N+1), the last node index is N.
    input_node = 0
    output_node = num_nodes - 1

    # Standard BFS setup
    queue = deque([input_node])
    visited = {input_node}

    while queue:
        current_node = queue.popleft()

        # Find all reachable neighbors from the current_node.
        # A connection from `current_node` to a layer whose node index is `neighbor_node` 
        # exists if A[current_node, neighbor_node] == 1.
        for neighbor_node in range(current_node + 1, num_nodes):
            if adjacency_matrix[current_node, neighbor_node] == 1:
                if neighbor_node not in visited:
                    visited.add(neighbor_node)
                    queue.append(neighbor_node)

    # After the traversal, the graph is connected if the final output node was visited.
    return output_node in visited
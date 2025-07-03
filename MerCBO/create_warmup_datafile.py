import torch
import numpy as np
import os  # <-- THIS LINE IS THE FIX
import pandas as pd

# --- You need to complete this part ---
# Find the minimum perplexity from each of the 16 warm-up run CSV files
# and paste them here in the SAME chronological order as your vectors.
PERPLEXITY_VALUES = [
    # Example:
    90.90708356355962,   # Perplexity for the arch with hash 45165...
    92.90126980214195,   # Perplexity for the arch with hash -37655...
    95.3705032431753,    # Perplexity for the arch with hash 17327...
    91.75682506098762,
    93.39140607375099,
    90.66656047217859,
    93.35462826986213,
    92.69950522225938,
    90.23932738719847,
    89.92063771192069,
    92.30181489313844,
    90.6196809043125,
    87.05781978918425,
    94.38390228565287,
    93.7263610662438,
    91.34457688458778
]
# ------------------------------------


# The 16 architecture vectors you provided
ARCHITECTURE_VECTORS = [
    [1,0,0,0,0,0,1,0,0,0,0,1,0,0,0,1,0,0,1,0,1,0,0],
    [1,0,0,0,0,0,1,0,0,0,0,1,0,0,0,1,0,0,1,0,1,0,1],
    [1,0,0,0,0,0,1,0,0,0,0,1,0,0,0,1,0,0,1,0,1,0,2],
    [1,0,0,0,0,0,1,0,0,0,0,1,0,0,0,1,0,0,1,0,1,0,3],
    [1,0,0,0,0,0,1,0,0,0,0,1,0,0,0,1,0,0,1,0,1,1,0],
    [1,0,0,0,0,0,1,0,0,0,0,1,0,0,0,1,0,0,1,0,1,1,1],
    [1,0,0,0,0,0,1,0,0,0,0,1,0,0,0,1,0,0,1,0,1,1,2],
    [1,0,0,0,0,0,1,0,0,0,0,1,0,0,0,1,0,0,1,0,1,1,3],
    [1,0,0,0,0,0,1,0,0,0,0,1,0,0,0,1,0,0,1,0,1,2,0],
    [1,0,0,0,0,0,1,0,0,0,0,1,0,0,0,1,0,0,1,0,1,2,1],
    [1,0,0,0,0,0,1,0,0,0,0,1,0,0,0,1,0,0,1,0,1,2,2],
    [1,0,0,0,0,0,1,0,0,0,0,1,0,0,0,1,0,0,1,0,1,2,3],
    [1,0,0,0,0,0,1,0,0,0,0,1,0,0,0,1,0,0,1,0,1,3,0],
    [1,0,0,0,0,0,1,0,0,0,0,1,0,0,0,1,0,0,1,0,1,3,1],
    [1,0,0,0,0,0,1,0,0,0,0,1,0,0,0,1,0,0,1,0,1,3,2],
    [1,0,0,0,0,0,1,0,0,0,0,1,0,0,0,1,0,0,1,0,1,3,3]
]

def package_data():
    if len(PERPLEXITY_VALUES) != len(ARCHITECTURE_VECTORS):
        print("Error: The number of perplexity values must match the number of vectors.")
        return

    # Convert the Python lists into the correct PyTorch tensors
    eval_inputs = torch.tensor(ARCHITECTURE_VECTORS, dtype=torch.long)
    
    # Remember to negate the perplexity for MerCBO, which maximizes
    eval_outputs = torch.tensor([[-p] for p in PERPLEXITY_VALUES], dtype=torch.float32)
    
    output_dir = "./warmup_data"
    output_file = os.path.join(output_dir, "warmup_results.pt")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the data to a file
    torch.save({
        'eval_inputs': eval_inputs,
        'eval_outputs': eval_outputs,
    }, output_file)

    print(f"Successfully packaged {eval_inputs.shape[0]} warm-up results into:")
    print(output_file)
    print(f"Input tensor shape: {eval_inputs.shape}")
    print(f"Output tensor shape: {eval_outputs.shape}")

if __name__ == '__main__':
    package_data()
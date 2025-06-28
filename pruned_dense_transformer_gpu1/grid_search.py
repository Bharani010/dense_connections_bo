# -*- coding: utf-8 -*-

from config import get_config
from train import train_model

# the empty [] means the encoder is not getting any skip connections.
# the[#x] with number inside it means that it is getting output as a skip connection from that particular layer
# we are considering the previous output layers as it in the encoder class in dmodel_ense.enc file.
dense_connections = {
    "E1":   [[0], [], [1], [1], [1], [1]],        
    "E2":   [[0], [], [], [2], [2], [2]],        
    "E3":   [[0], [], [], [], [3], [3]],     
    "E4":   [[0], [], [], [], [], [4]],
    "E12":  [[0], [], [1], [1, 2], [1, 2], [1, 2]],
    "E23":  [[0], [], [], [2], [2, 3], [2, 3]],   
    "E34":  [[0], [], [], [], [3], [3,4]],         
    "E123": [[0], [], [1], [1,2], [1,2,3], [1,2,3]],        
    "E234": [[0], [], [], [2], [2,3], [2,3,4]],       
    "E1234": [[0], [], [1], [1, 2], [1,2,3], [1,2,3,4]],
}

def run_grid_search():
    d_models_list = [512]
    num_heads_list = [2, 4,8,16]
    connection_patterns = list(dense_connections.keys())  

    # 4 d_models × 4 num_heads × 10 connection patterns = 160 total runs
    for d_model in d_models_list:
        for num_heads in num_heads_list:
            for conn_type in connection_patterns:
                # Get the base configuration
                config = get_config()

                # Remove any ambiguity by using only "d_models" for the dense model.
                config["d_models"] = [d_model]
                if "d_model" in config:
                    del config["d_model"]

                config["num_heads"] = num_heads

                # Add the connection pattern to the config
                config["dense_connections"] = dense_connections[conn_type]

                # Update folder and experiment naming so that each run saves to a unique file.
                config["model_folder"] = f"emb{d_model}_head{num_heads}_{conn_type}_{config['lang_src']}_{config['lang_tgt']}_{config['net_name']}"
                config["experiment_name"] = f"runs/tmodel_emb{d_model}_head{num_heads}_{conn_type}"

                # Set preload to None so each experiment starts fresh.
                config["preload"] = None

                print(f"Starting experiment with d_model={d_model}, num_heads={num_heads}, connection={conn_type}")
                train_model(config)
                print(f"Finished experiment with d_model={d_model}, num_heads={num_heads}, connection={conn_type}\n")

if __name__ == '__main__':
    run_grid_search()
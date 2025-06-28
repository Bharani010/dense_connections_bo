# -*- coding: utf-8 -*-

from pathlib import Path

def get_config():
    return {
        "net_name": "model_dense_enc",
        "batch_size": 8,
        "num_epochs": 15,
        "lr": 10**-4,
        "seq_len": 350,
        "d_models": [512],	
        "num_layers": 6,
        "num_heads": 2,
        "datasource": 'opus_books',
        "lang_src": "en",
        "lang_tgt": "it",
        "model_folder": "emb{d_model}_head{num_heads}_{lang_src}_{lang_tgt}_{net_name}",
        "model_basename": "tmodel_",
        "preload": "latest",
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel"
    }

def get_weights_file_path(config, epoch: str):
    # Format the model_folder string using the values from config
    model_folder = f"{config['datasource']}_{config['model_folder'].format(d_model=config['d_models'][0], num_heads=config['num_heads'], lang_src=config['lang_src'], lang_tgt=config['lang_tgt'], net_name=config['net_name'])}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)


# Find the latest weights file in the weights folder
def latest_weights_file_path(config):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])

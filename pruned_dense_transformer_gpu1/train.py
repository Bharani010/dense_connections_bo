# -*- coding: utf-8 -*-
from model_dense_enc import build_transformer
from dataset import BilingualDataset, causal_mask
from config import get_config, get_weights_file_path, latest_weights_file_path

#import torchtext.datasets as datasets
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import LambdaLR


import warnings
import numpy as np
from tqdm import tqdm
import os
import csv
from pathlib import Path

# Huggingface datasets and tokenizers
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

import torchmetrics
from torch.utils.tensorboard import SummaryWriter

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device,connections_matrix):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source, source_mask, connections_matrix)
    # Initialize the decoder input the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        # build mask for target
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # calculate output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # get next token
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)

def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        # Most code taken from: https://huggingface.co/docs/tokenizers/quicktour
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer
    
# Modify get_ds to filter sentences longer than seq_len
def get_ds(config):
    # It only has the train split, so we divide it ourselves
    ds_raw = load_dataset('opus_books', f"{config['lang_src']}-{config['lang_tgt']}", split='train')

    # Filter out sentences longer than seq_len
    ds_filtered = [item for item in ds_raw if len(item['translation'][config['lang_src']]) <= config['seq_len'] and len(item['translation'][config['lang_tgt']]) <= config['seq_len']]
    # Limit to 1/4 of the dataset
    ds_filtered = ds_filtered[:len(ds_filtered) ]


    # Build tokenizers
    tokenizer_src = get_or_build_tokenizer(config, ds_filtered, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_filtered, config['lang_tgt'])

    # Keep 90% for training, 10% for validation
    train_ds_size = int(0.9 * len(ds_filtered))
    val_ds_size = len(ds_filtered) - train_ds_size
    print("length_of_dataset :" + str(len(ds_filtered)))

    train_ds_raw, val_ds_raw = random_split(ds_filtered, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    # Find the maximum length of each sentence in the source and target sentence
    max_len_src = 0
    max_len_tgt = 0

    for item in ds_filtered:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')
    

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config["seq_len"], config["seq_len"])

    init_weights_path = "init_model.pth"

    if os.path.exists(init_weights_path):
        # Load the saved initial weights
        print(f"Loading initial weights from {init_weights_path}")
        model.load_state_dict(torch.load(init_weights_path))
    else:
        print(f"Saving initial weights at {init_weights_path}")
        torch.save(model.state_dict(), init_weights_path)

    return model



import math

# Function to calculate perplexity
def calculate_perplexity(loss):
    return math.exp(loss)

def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, writer, connections_matrix, num_examples=2):
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []
    total_loss = 0.0
    num_batches = 0

    try:
        # get the console window width
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        # If we can't get the console width, use 80 as default
        console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            num_batches += 1
            encoder_input = batch["encoder_input"].to(device) # (b, seq_len)
            encoder_mask = batch["encoder_mask"].to(device) # (b, 1, 1, seq_len)

            # check that the batch size is 1
            assert encoder_input.size(
                0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device, connections_matrix)

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)
            
            # Compute loss for perplexity calculation
            decoder_input = batch['decoder_input'].to(device) # (B, seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # (B, 1, seq_len, seq_len)
            label = batch['label'].to(device) # (B, seq_len)

            encoder_output = model.encode(encoder_input, encoder_mask, connections_matrix) # (B, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (B, seq_len, d_model)
            proj_output = model.project(decoder_output) # (B, seq_len, vocab_size)

            loss = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1)(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            total_loss += loss.item()


            # Limit printing but not the loss calculation
            if count <= num_examples:
                # Print the source, target, and model output
                print_msg('-' * console_width)
                print_msg(f"{f'SOURCE: ':>12}{source_text}")
                print_msg(f"{f'TARGET: ':>12}{target_text}")
                print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print_msg('-' * console_width)
                
                
    if writer:
        # Evaluate the character error rate
        # Compute the char error rate 
        metric = torchmetrics.CharErrorRate()
        cer = metric(predicted, expected)
        writer.add_scalar('validation cer', cer, global_step)
        writer.flush()

        # Compute the word error rate
        metric = torchmetrics.WordErrorRate()
        wer = metric(predicted, expected)
        writer.add_scalar('validation wer', wer, global_step)
        writer.flush()

        # Compute the BLEU metric
        metric = torchmetrics.BLEUScore()
        bleu = metric(predicted, expected)
        writer.add_scalar('validation BLEU', bleu, global_step)
        writer.flush()

        # Compute and log the validation perplexity
        avg_loss = total_loss / num_batches
        perplexity = calculate_perplexity(avg_loss)
        writer.add_scalar('validation perplexity', perplexity, global_step)
        writer.flush()
        
    return total_loss / num_batches

def train_model(config):
    # Define the device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)
    if (device == 'cuda'):
        print(f"Device name: {torch.cuda.get_device_name(device.index)}")
        print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")
    elif (device == 'mps'):
        print(f"Device name: <mps>")
    else:
        print("NOTE: If you have a GPU, consider using it for training.")
        print("      On a Windows machine with NVidia GPU, check this video: https://www.youtube.com/watch?v=GMSjDTU8Zlc")
        print("      On a Mac machine, run: pip3 install --pre torch torchvision torchaudio torchtext --index-url https://download.pytorch.org/whl/nightly/cpu")
    device = torch.device(device)

    # CHANGE: Define a default connections_matrix for running train.py standalone.
    # This represents a standard sequential encoder.
    N = config['num_layers']
    default_matrix = np.zeros((N + 1, N + 1), dtype=int)
    for i in range(N):
        # This creates connections from node i to layer i+1 (sequential flow)
        # In our graph representation, this is a connection from node i to node i+1.
        default_matrix[i, i + 1] = 1
    
    # Get the connections_matrix from the config, or use the default sequential one.
    # The Bayesian Optimization wrapper will place the matrix it wants to test into the config dict.
    connections_matrix = config.get("connections_matrix", default_matrix)
    # Convert to a tensor and move to the correct device for the model
    connections_matrix = torch.tensor(connections_matrix, device=device, dtype=torch.int)


    # Make sure the weights folder exists
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    Path(model_folder).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    # Tensorboard
    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    # If the user specified a model to preload before training, load it
    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    # NOTE: Preloading weights is tricky with dynamically changing architectures.
    # It's recommended to keep preload=None when using the BO search.
    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print('No model to preload, starting from scratch')

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)
    
    avg_loss = 0.0

    # Open CSV file for writing perplexity values before the loop starts
    csv_path = os.path.join(model_folder, f"{config['model_folder']}.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, mode='w', newline='') as file:
        writer_csv = csv.writer(file)
        writer_csv.writerow(['Epoch', 'Validation Perplexity'])
    
        for epoch in range(initial_epoch, config['num_epochs']):
            torch.cuda.empty_cache()
            model.train()
            batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
            
            for batch in batch_iterator:
    
                encoder_input = batch['encoder_input'].to(device) # (b, seq_len)
                decoder_input = batch['decoder_input'].to(device) # (B, seq_len)
                encoder_mask = batch['encoder_mask'].to(device) # (B, 1, 1, seq_len)
                decoder_mask = batch['decoder_mask'].to(device) # (B, 1, seq_len, seq_len)
    
                # CHANGE: Pass the connections_matrix to the model's encode method
                # Run the tensors through the encoder, decoder and the projection layer
                encoder_output = model.encode(encoder_input, encoder_mask, connections_matrix) # (B, seq_len, d_model)
                decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (B, seq_len, d_model)
                proj_output = model.project(decoder_output) # (B, seq_len, vocab_size)
    
                # Compare the output with the label
                label = batch['label'].to(device) # (B, seq_len)
    
                # Compute the loss using a simple cross entropy
                loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
                batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})
    
                # Log the loss and perplexity
                writer.add_scalar('train loss', loss.item(), global_step)
                writer.add_scalar('train perplexity', calculate_perplexity(loss.item()), global_step)
                writer.flush()
    
                # Backpropagate the loss
                loss.backward()
    
                # Update the weights
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
    
                global_step += 1
    
            # CHANGE: Pass the connections_matrix to the validation function
            # Run validation at the end of every epoch
            avg_loss = run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer, connections_matrix)
    
            # Compute and log the validation perplexity
            avg_perplexity = calculate_perplexity(avg_loss)
            
            # Write perplexity to CSV after each epoch
            writer_csv.writerow([epoch, avg_perplexity])
            file.flush()  # Ensure data is written to the file
            
            # Print the perplexity for the current epoch
            print(f'Epoch {epoch:02d}, Validation Perplexity: {avg_perplexity:.4f}')
    
            # Save the model at the end of every epoch
            # Note: The saved model state does not include the architecture.
            # You would need the connections_matrix to rebuild this exact model later.
            model_filename = get_weights_file_path(config, f"{epoch:02d}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'global_step': global_step
            }, model_filename)

    final_avg_loss = avg_loss
    final_perplexity = calculate_perplexity(final_avg_loss)
    print(f'final validation perplexity: {final_perplexity}')
    
    #added this so MerCBO can get the scaler back
    return float(final_perplexity)
#    # Save the final_perplexity to a file
#    with open('final_perplexity_basic_ep100_6encdec_emb256.txt', 'w') as file:
#        file.write(f'final validation perplexity: {final_perplexity}\n')

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)
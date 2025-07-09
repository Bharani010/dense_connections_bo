# File: /home/sveerepa/dense_connections_bo/main.py

import sys
import time
import argparse
import os
import torch
import numpy as np

# MerCBO Library Imports
from MerCBO.graphGP.kernels.diffusionkernel import DiffusionKernel
from MerCBO.graphGP.models.gp_regression import GPRegression
from MerCBO.graphGP.sampler.sample_posterior import posterior_sampling
from MerCBO.acquisition.acquisition_optimization import next_evaluation_with_thompson_sampling
from MerCBO.utils import model_data_filenames, load_model_data, displaying_and_logging

# Note: The fix_numpy_aliases shim is an alternative to running the `sed` commands.
# If you have already patched the library files, this import is not strictly necessary.
try:
    import fix_numpy_aliases
except ModuleNotFoundError:
    pass # Ignore if not found


EXPERIMENTS_DIRECTORY = './MerCBO_experiments'

def MerCBO(objective=None, n_eval=200, path=None, parallel=False, store_data=True, device='cpu', **kwargs):
    """ Main MerCBO optimization loop. """
    assert (path is None) != (objective is None)
    
    if objective is not None:
        # Get problem definition from the objective class
        n_vertices = objective.n_vertices
        adj_mat_list = objective.adjacency_mat
        fourier_freq_list = objective.fourier_freq
        fourier_basis_list = objective.fourier_basis

        # --- LOGIC TO LOAD DATA OR RUN WARM-UPS ---
        if hasattr(objective, 'initial_outputs') and objective.initial_outputs is not None:
            print(f"INFO: [main.py] Found {objective.suggested_init.shape[0]} pre-computed evaluation points. Loading them directly.")
            eval_inputs = objective.suggested_init
            eval_outputs = objective.initial_outputs
        else:
            # Fallback to original behavior: run warm-ups from scratch
            print("INFO: [main.py] No pre-computed data found. Running warm-up evaluations from scratch.")
            eval_inputs = objective.suggested_init
            eval_outputs = torch.zeros(eval_inputs.size(0), 1)
            for i in range(eval_inputs.size(0)):
                print(f"--- Running Warm-up Evaluation {i+1}/{eval_inputs.size(0)} ---")
                eval_outputs[i] = objective.evaluate(eval_inputs[i])
        
        n_init = eval_inputs.size(0)
        
        # --- The rest of the setup proceeds ---
        exp_dir = EXPERIMENTS_DIRECTORY
        objective_name = '_'.join([objective.__class__.__name__])
        _, _, logfile_dir = model_data_filenames(exp_dir=exp_dir, objective_name=objective_name)

        eval_inputs, eval_outputs = eval_inputs.to(device), eval_outputs.to(device)
        
        grouped_log_beta = torch.ones(len(fourier_freq_list))
        kernel = DiffusionKernel(grouped_log_beta=grouped_log_beta, fourier_freq_list=fourier_freq_list, fourier_basis_list=fourier_basis_list)
        surrogate_model = GPRegression(kernel=kernel)
        
        surrogate_model.init_param(eval_outputs)
        print('(%s) Performing initial MCMC sampling (Burn-in phase)...' % time.strftime('%H:%M:%S', time.localtime()))
        sample_posterior = posterior_sampling(surrogate_model, eval_inputs, eval_outputs, n_vertices, adj_mat_list)
        log_beta, sorted_partition = sample_posterior[0][0], sample_posterior[1][0]
        
        time_list = [time.time()] * n_init
        elapse_list, pred_mean_list, pred_std_list, pred_var_list = [0] * n_init, [0] * n_init, [0] * n_init, [0] * n_init
    
    else: # Logic for resuming from a MerCBO checkpoint
        surrogate_model, cfg_data, logfile_dir = load_model_data(path, exp_dir=EXPERIMENTS_DIRECTORY)


    # Main BO loop for n_eval iterations
    for eval_idx in range(n_eval):
        start_time = time.time()
        print(f"\n--- Starting Bayesian Optimization Step {eval_idx + 1}/{n_eval} ---")
        reference = torch.min(eval_outputs, dim=0)[0].item()
        print('(%s) Sampling from GP Posterior...' % time.strftime('%H:%M:%S', time.localtime()))
        sample_posterior = posterior_sampling(surrogate_model, eval_inputs, eval_outputs, n_vertices, adj_mat_list,
                                              log_beta, sorted_partition, n_sample=10, n_burn=0, n_thin=1)
        hyper_samples, log_beta_samples, partition_samples, freq_samples, basis_samples, _ = sample_posterior
        log_beta = log_beta_samples[-1]
        sorted_partition = partition_samples[-1]
        
        inference_samples = inference_sampling(eval_inputs, eval_outputs, n_vertices,
                                               hyper_samples, log_beta_samples, partition_samples,
                                               freq_samples, basis_samples)
        
        suggestion = next_evaluation_with_thompson_sampling(eval_inputs, eval_outputs, inference_samples, partition_samples, hyper_samples[-1][-1], n_vertices, log_beta)
        next_eval, pred_mean, pred_std, pred_var = suggestion

        print('(%s) Evaluating suggested architecture...' % time.strftime('%H:%M:%S', time.localtime()))
        eval_inputs = torch.cat([eval_inputs, next_eval.view(1, -1)], 0)
        eval_outputs = torch.cat([eval_outputs, objective.evaluate(eval_inputs[-1]).view(1, 1)])
        
        time_list.append(time.time())
        elapse_list.append(time.time() - start_time)
        pred_mean_list.append(pred_mean.item())
        pred_std_list.append(pred_std.item())
        pred_var_list.append(pred_var.item())

        displaying_and_logging(logfile_dir, eval_inputs, eval_outputs, pred_mean_list, pred_std_list, pred_var_list,
                               time_list, elapse_list, hyper_samples, log_beta_samples, store_data)


if __name__ == '__main__':
    # Import your custom objective classes
    from MerCBO.experiments.test_functions.transformer_perplexity import TransformerPerplexity
    from MerCBO.experiments.test_functions.EncoderBO import EncoderBO

    # Setup the argument parser
    parser_ = argparse.ArgumentParser(description='MerCBO : Mercer Features for Efficient Combinatorial Bayesian Optimization')
    parser_.add_argument('--n_eval', type=int, default=1)
    parser_.add_argument('--path', type=str, default=None)
    parser_.add_argument('--objective', type=str, required=True)
    parser_.add_argument('--device', type=str, default='cuda:1')
    parser_.add_argument('--load_initial_data', type=str, default=None, help='Path to a .pt file with pre-computed eval_inputs and eval_outputs.')

    args_ = parser_.parse_args()
    kwag_ = vars(args_)
    objective_name = kwag_.pop('objective')

    # --- Logic to correctly create and pass the objective ---
    initial_data = None
    if args_.load_initial_data and os.path.exists(args_.load_initial_data):
        print(f"INFO: [main.py] Found initial data path: {args_.load_initial_data}")
        initial_data = torch.load(args_.load_initial_data)

    if objective_name == 'nas_encoder':
        print("INFO: [main.py] Initializing 'nas_encoder' objective.")
        objective_instance = EncoderBO(initial_data=initial_data)
        kwag_['objective'] = objective_instance
        MerCBO(**kwag_)
    elif objective_name == 'transformer':
        print("INFO: [main.py] Initializing 'transformer' objective.")
        objective_instance = TransformerPerplexity() # This older wrapper doesn't use initial_data
        kwag_['objective'] = objective_instance
        MerCBO(**kwag_)
    else:
        # This part handles the original MerCBO benchmarks like 'ising' and 'labs'
        print(f"INFO: [main.py] Initializing benchmark objective '{objective_name}'.")
        # Add ising/labs logic back in here if you need it, otherwise raise error.
        raise NotImplementedError(f"Benchmark objective '{objective_name}' not configured in this script.")

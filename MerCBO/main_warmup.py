import sys
import time
import argparse
import os # Make sure os is imported for os.path.exists

import torch
import numpy as np

from MerCBO.graphGP.kernels.diffusionkernel import DiffusionKernel
from MerCBO.graphGP.models.gp_regression import GPRegression
from MerCBO.graphGP.sampler.sample_posterior import posterior_sampling
from MerCBO.acquisition.acquisition_optimization import next_evaluation_with_thompson_sampling
from MerCBO.utils import model_data_filenames, load_model_data, displaying_and_logging

# Note: The 'fix_numpy_aliases' shim is no longer needed if you have already
# patched the MerCBO library files with the `sed` commands to replace np.long etc.
# If you haven't, you can keep the import.
# import fix_numpy_aliases

EXPERIMENTS_DIRECTORY = './MerCBO_experiments' # Use a local experiments directory


def MerCBO(objective=None, n_eval=200, path=None, parallel=False, store_data=True, **kwargs):
    """
    Main MerCBO optimization loop.
    """
    assert (path is None) != (objective is None)

    n_vertices = adj_mat_list = None
    eval_inputs = eval_outputs = log_beta = sorted_partition = None
    time_list = elapse_list = pred_mean_list = pred_std_list = pred_var_list = None

    if objective is not None:
        # --- Setup logging and get problem definition from the objective class ---
        exp_dir = EXPERIMENTS_DIRECTORY
        objective_name = '_'.join([objective.__class__.__name__])
        model_filename, data_cfg_filaname, logfile_dir = model_data_filenames(exp_dir=exp_dir,
                                                                              objective_name=objective_name)
        n_vertices = objective.n_vertices
        adj_mat_list = objective.adjacency_mat
        fourier_freq_list = objective.fourier_freq
        fourier_basis_list = objective.fourier_basis
        
        # --- NEW LOGIC: Use pre-computed data if available in the objective ---
        if hasattr(objective, 'initial_outputs') and objective.initial_outputs is not None:
            print(f"main.py: Found {objective.suggested_init.shape[0]} pre-computed data points from objective. Loading them directly.")
            eval_inputs = objective.suggested_init
            eval_outputs = objective.initial_outputs
            n_init = eval_inputs.size(0)
            # We completely skip the expensive warm-up evaluation loop
        else:
            # Fallback to original behavior: run warm-ups from scratch
            print("main.py: No pre-computed data found. Running warm-up evaluations from scratch.")
            eval_inputs = objective.suggested_init
            n_init = eval_inputs.size(0)
            eval_outputs = torch.zeros(n_init, 1)
            # This loop runs the warm-up evaluations
            for i in range(n_init):
                print(f"--- Running Warm-up Evaluation {i+1}/{n_init} ---")
                eval_outputs[i] = objective.evaluate(eval_inputs[i])
        
        # Move tensors to the correct device specified by the user
        device = kwargs.get('device', 'cpu')
        eval_inputs = eval_inputs.to(device)
        eval_outputs = eval_outputs.to(device)

        # --- The rest of the setup proceeds as before ---
        grouped_log_beta = torch.ones(len(fourier_freq_list))
        kernel = DiffusionKernel(grouped_log_beta=grouped_log_beta,
                                 fourier_freq_list=fourier_freq_list, fourier_basis_list=fourier_basis_list)
        surrogate_model = GPRegression(kernel=kernel)
        
        # Initialize other variables for logging
        time_list = [time.time()] * n_init
        elapse_list = [0] * n_init
        # We don't have predictions for pre-loaded points, so initialize with 0
        pred_mean_list = [0] * n_init
        pred_std_list = [0] * n_init
        pred_var_list = [0] * n_init

        # Initialize the GP model with the (loaded or newly computed) data
        surrogate_model.init_param(eval_outputs)
        print('(%s) Performing initial MCMC sampling (Burn-in phase)...' % time.strftime('%H:%M:%S', time.localtime()))
        sample_posterior = posterior_sampling(surrogate_model, eval_inputs, eval_outputs, n_vertices, adj_mat_list,
                                              n_sample=1, n_burn=99, n_thin=1)
        log_beta = sample_posterior[0][0] # Corrected index based on typical sampler output
        sorted_partition = sample_posterior[1][0]
        print('')

    else: # This block is for resuming from a MerCBO checkpoint, not used in our flow
        surrogate_model, cfg_data, logfile_dir = load_model_data(path, exp_dir=EXPERIMENTS_DIRECTORY)

    # The main BO loop for n_eval iterations
    for eval_idx in range(n_eval):
        start_time = time.time()
        print(f"\n--- Starting Bayesian Optimization Step {eval_idx + 1}/{n_eval} ---")
        # ... (The rest of the BO loop is the same) ...
        reference = torch.min(eval_outputs, dim=0)[0].item()
        print('(%s) Sampling from GP Posterior...' % time.strftime('%H:%M:%S', time.localtime()))
        sample_posterior = posterior_sampling(surrogate_model, eval_inputs, eval_outputs, n_vertices, adj_mat_list,
                                              log_beta, sorted_partition, n_sample=10, n_burn=0, n_thin=1)
        hyper_samples, log_beta_samples, partition_samples, freq_samples, basis_samples, edge_mat_samples = sample_posterior
        log_beta = log_beta_samples[-1]
        sorted_partition = partition_samples[-1]
        
        inference_samples = inference_sampling(eval_inputs, eval_outputs, n_vertices,
                                               hyper_samples, log_beta_samples, partition_samples,
                                               freq_samples, basis_samples)
        
        suggestion = next_evaluation_with_thompson_sampling(eval_inputs, eval_outputs, inference_samples, partition_samples, hyper_samples[-1][-1], n_vertices, log_beta)
        next_eval, pred_mean, pred_std, pred_var = suggestion

        processing_time = time.time() - start_time

        print('(%s) Evaluating suggested architecture...' % time.strftime('%H:%M:%S', time.localtime()))
        eval_inputs = torch.cat([eval_inputs, next_eval.view(1, -1)], 0)
        eval_outputs = torch.cat([eval_outputs, objective.evaluate(eval_inputs[-1]).view(1, 1)])
        assert not torch.isnan(eval_outputs).any()

        time_list.append(time.time())
        elapse_list.append(processing_time)
        pred_mean_list.append(pred_mean.item())
        pred_std_list.append(pred_std.item())
        pred_var_list.append(pred_var.item())

        displaying_and_logging(logfile_dir, eval_inputs, eval_outputs, pred_mean_list, pred_std_list, pred_var_list,
                               time_list, elapse_list, hyper_samples, log_beta_samples, store_data)
        print('Optimizing %s with regularization %.2E up to %4d...'
              % (objective.__class__.__name__, 0, n_eval))


if __name__ == '__main__':
    # Import your custom objective classes
    from MerCBO.experiments.test_functions.transformer_perplexity import TransformerPerplexity
    from MerCBO.experiments.test_functions.EncoderBO import EncoderBO

    # Setup the argument parser
    parser_ = argparse.ArgumentParser(description='MerCBO : Mercer Features for Efficient Combinatorial Bayesian Optimization')
    parser_.add_argument('--n_eval', type=int, default=1)
    parser_.add_argument('--path', type=str, default=None)
    parser_.add_argument('--objective', type=str, required=True)
    parser_.add_argument('--device', type=int, default=None)
    # ADDED: New argument to load pre-computed data
    parser_.add_argument('--load_initial_data', type=str, default=None, help='Path to a .pt file with pre-computed eval_inputs and eval_outputs.')

    args_ = parser_.parse_args()
    kwag_ = vars(args_)
    objective_ = kwag_.pop('objective') # Pop so it's not passed to MerCBO function via kwargs

    # --- Custom objective handling ---
    initial_data_to_pass = None
    if args_.load_initial_data and os.path.exists(args_.load_initial_data):
        print(f"[main.py] Found initial data path: {args_.load_initial_data}")
        initial_data_to_pass = torch.load(args_.load_initial_data)

    if objective_ == 'transformer':
        print("[main.py] INFO: Initializing 'transformer' objective (10 fixed patterns).")
        kwag_['objective'] = TransformerPerplexity() # Does not use initial_data
        MerCBO(**kwag_)
    elif objective_ == 'nas_encoder':
        print("[main.py] INFO: Initializing 'nas_encoder' objective (full architecture search).")
        # Pass the loaded data to the constructor
        kwag_['objective'] = EncoderBO(initial_data=initial_data_to_pass)
        MerCBO(**kwag_)
    else:
        raise NotImplementedError(f"Objective '{objective_}' not recognized.")

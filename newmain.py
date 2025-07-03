import sys
import time
import argparse
import os
import torch
import numpy as np

# Import all necessary modules from the MerCBO library
from MerCBO.graphGP.kernels.diffusionkernel import DiffusionKernel
from MerCBO.graphGP.models.gp_regression import GPRegression
from MerCBO.graphGP.sampler.sample_posterior import posterior_sampling
from MerCBO.acquisition.acquisition_optimization import next_evaluation_with_thompson_sampling
from MerCBO.utils import model_data_filenames, load_model_data, displaying_and_logging

# This is a one-time patch for deprecated numpy aliases in the library
# It's better to run the sed commands once, but this also works.
try:
    import fix_numpy_aliases
except ModuleNotFoundError:
    print("Warning: fix_numpy_aliases.py not found. Some numpy functions might be deprecated.")


EXPERIMENTS_DIRECTORY = './MerCBO_experiments'

def MerCBO(objective=None, n_eval=200, path=None, parallel=False, store_data=True, device='cpu', **kwargs):
    """ Main MerCBO optimization loop. """
    assert (path is None) != (objective is None)
    
    # --- This block handles the setup, including loading pre-computed data ---
    if objective is not None:
        # Get problem definition from the objective class
        n_vertices = objective.n_vertices
        adj_mat_list = objective.adjacency_mat
        fourier_freq_list = objective.fourier_freq
        fourier_basis_list = objective.fourier_basis

        # Check if the objective was initialized with pre-computed data
        if hasattr(objective, 'initial_outputs') and objective.initial_outputs is not None:
            print(f"INFO: Found {objective.suggested_init.shape[0]} pre-computed evaluation points. Loading them directly.")
            eval_inputs = objective.suggested_init
            eval_outputs = objective.initial_outputs
        else:
            # Fallback: run warm-ups from scratch if no data was loaded
            print("INFO: No pre-computed data found. Running warm-up evaluations from scratch.")
            eval_inputs = objective.suggested_init
            eval_outputs = torch.zeros(eval_inputs.size(0), 1)
            for i in range(eval_inputs.size(0)):
                print(f"--- Running Warm-up Evaluation {i+1}/{eval_inputs.size(0)} ---")
                eval_outputs[i] = objective.evaluate(eval_inputs[i])

        n_init = eval_inputs.size(0)
        
        # --- The rest of the setup proceeds, now with the correct initial data ---
        exp_dir = EXPERIMENTS_DIRECTORY
        objective_name = '_'.join([objective.__class__.__name__])
        _, _, logfile_dir = model_data_filenames(exp_dir=exp_dir, objective_name=objective_name)

        eval_inputs, eval_outputs = eval_inputs.to(device), eval_outputs.to(device)
        
        grouped_log_beta = torch.ones(len(fourier_freq_list))
        kernel = DiffusionKernel(grouped_log_beta=grouped_log_beta, fourier_freq_list=fourier_freq_list, fourier_basis_list=fourier_basis_list)
        surrogate_model = GPRegression(kernel=kernel)
        
        surrogate_model.init_param(eval_outputs)
        print('(%s) Performing initial MCMC sampling (Burn-in phase)...' % time.strftime('%H:%M:%S', time.localtime()))
        sample_posterior = posterior_sampling(surrogate_model, eval_inputs, eval_outputs, n_vertices, adj_mat_list, n_sample=1, n_burn=99, n_thin=1)
        log_beta, sorted_partition = sample_posterior[0][0], sample_posterior[1][0]
        
        time_list = [time.time()] * n_init
        elapse_list = [0] * n_init
        pred_mean_list, pred_std_list, pred_var_list = [0] * n_init, [0] * n_init, [0] * n_init
    
    # ... (The rest of the MerCBO function and BO loop is the same) ...
    # ... (For brevity, only the setup part is shown, the loop logic doesn't need changes) ...
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
    from MerCBO.experiments.test_functions.EncoderBO import EncoderBO

    parser_ = argparse.ArgumentParser(description='MerCBO : Mercer Features for Efficient Combinatorial Bayesian Optimization')
    parser_.add_argument('--n_eval', type=int, default=1)
    parser_.add_argument('--path', type=str, default=None)
    parser_.add_argument('--objective', type=str, required=True)
    parser_.add_argument('--device', type=str, default='cpu') # CHANGED: default to cpu, can be 'cuda:0', 'cuda:1' etc.
    parser_.add_argument('--load_initial_data', type=str, default=None)

    args_ = parser_.parse_args()
    kwag_ = vars(args_)
    objective_name = kwag_.pop('objective')

    # --- This logic correctly creates and passes the objective ---
    if objective_name == 'nas_encoder':
        print("[main.py] INFO: Initializing 'nas_encoder' objective (full architecture search).")
        initial_data = None
        if args_.load_initial_data and os.path.exists(args_.load_initial_data):
            print(f"Found initial data path: {args_.load_initial_data}")
            initial_data = torch.load(args_.load_initial_data)
        
        # Pass the loaded data to the constructor
        objective_instance = EncoderBO(initial_data=initial_data)
        kwag_['objective'] = objective_instance
        MerCBO(**kwag_)
    else:
        raise NotImplementedError(f"Objective '{objective_name}' not recognized.")

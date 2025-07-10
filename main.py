import fix_numpy_aliases  # for numpy import fix
import sys
import time
import argparse
import os
import torch
import numpy as np

from MerCBO.graphGP.kernels.diffusionkernel import DiffusionKernel
from MerCBO.graphGP.models.gp_regression import GPRegression
from MerCBO.graphGP.sampler.sample_posterior import posterior_sampling

from MerCBO.acquisition.acquisition_optimization import next_evaluation_with_thompson_sampling
from MerCBO.acquisition.acquisition_marginalization import inference_sampling
from MerCBO.utils import model_data_filenames, load_model_data, displaying_and_logging

# Import all objectives
from MerCBO.experiments.random_seed_config import generate_random_seed_pair_ising
from MerCBO.experiments.test_functions.binary_categorical import Ising
from MerCBO.experiments.test_functions.labs import LABS_OBJ
from MerCBO.experiments.test_functions.transformer_perplexity import TransformerPerplexity
from MerCBO.experiments.test_functions.EncoderBO import EncoderBO


EXPERIMENTS_DIRECTORY = '../MerCBO_experiments'


def MerCBO(objective=None, n_eval=200, path=None, parallel=False, store_data=True, **kwargs):
    """
    Main MerCBO optimization loop.
    """
    assert (path is None) != (objective is None)

    # --- This block handles the main setup based on the objective ---
    if objective is not None:
        # Get problem definition from the objective class
        n_vertices = objective.n_vertices
        adj_mat_list = objective.adjacency_mat
        fourier_freq_list = objective.fourier_freq
        fourier_basis_list = objective.fourier_basis

        # THIS IS THE KEY CHANGE: Check if the objective has pre-computed outputs
        if hasattr(objective, 'initial_outputs') and objective.initial_outputs is not None:
            print("INFO: [MerCBO] Loading pre-computed warm-up data from objective.")
            eval_inputs = objective.suggested_init
            eval_outputs = objective.initial_outputs
        else:
            # If not, run the evaluations from scratch (the 30+ hour process)
            print("INFO: [MerCBO] No pre-computed data found. Evaluating warm-up points now.")
            eval_inputs = objective.suggested_init
            eval_outputs = torch.zeros(eval_inputs.size(0), 1, device=eval_inputs.device)
            for i in range(eval_inputs.size(0)):
                print(f"--- Running Warm-up Evaluation {i+1}/{eval_inputs.size(0)} ---")
                eval_outputs[i] = objective.evaluate(eval_inputs[i])

        n_init = eval_inputs.size(0)

        # Create kernel and model
        grouped_log_beta = torch.ones(len(fourier_freq_list))
        kernel = DiffusionKernel(grouped_log_beta=grouped_log_beta,
                                 fourier_freq_list=fourier_freq_list, fourier_basis_list=fourier_basis_list)
        surrogate_model = GPRegression(kernel=kernel)

        # Initialize BO state
        log_beta = eval_outputs.new_zeros(eval_inputs.size(1))
        sorted_partition = [[m] for m in range(eval_inputs.size(1))]

        time_list = [time.time()] * n_init
        elapse_list = [0] * n_init
        pred_mean_list = [0] * n_init
        pred_std_list = [0] * n_init
        pred_var_list = [0] * n_init

        # Initialize the surrogate model and perform MCMC burn-in
        surrogate_model.init_param(eval_outputs)
        print('(%s) Performing initial MCMC sampling (Burn-in phase)...' % time.strftime('%H:%M:%S', time.localtime()))
        sample_posterior = posterior_sampling(surrogate_model, eval_inputs, eval_outputs, n_vertices, adj_mat_list,
                                              log_beta, sorted_partition, n_sample=1, n_burn=99, n_thin=1)
        log_beta = sample_posterior[1][0]
        sorted_partition = sample_posterior[2][0]
        print('')

        # Setup logging
        exp_dir = EXPERIMENTS_DIRECTORY
        objective_name = '_'.join([objective.__class__.__name__])
        _, _, logfile_dir = model_data_filenames(exp_dir=exp_dir, objective_name=objective_name)

    else:
        # This block is for resuming from a previous MerCBO run, not relevant to warm-up data
        surrogate_model, cfg_data, logfile_dir = load_model_data(path, exp_dir=EXPERIMENTS_DIRECTORY)
        # (unpack cfg_data...)

    # --- Main Bayesian Optimization Loop ---
    for _ in range(n_eval):
        start_time = time.time()
        print('(%s) Sampling from posterior...' % time.strftime('%H:%M:%S', time.localtime()))
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

        eval_inputs = torch.cat([eval_inputs, next_eval.view(1, -1)], 0)
        eval_outputs = torch.cat([eval_outputs, objective.evaluate(eval_inputs[-1]).view(1, 1)])
        assert not torch.isnan(eval_outputs).any()

        time_list.append(time.time())
        elapse_list.append(time.time() - start_time)
        pred_mean_list.append(pred_mean.item())
        pred_std_list.append(pred_std.item())
        pred_var_list.append(pred_var.item())

        displaying_and_logging(logfile_dir, eval_inputs, eval_outputs, pred_mean_list, pred_std_list, pred_var_list,
                               time_list, elapse_list, hyper_samples, log_beta_samples, store_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='MerCBO: Mercer Features for Efficient Combinatorial Bayesian Optimization')
    parser.add_argument('--n_eval', type=int, default=50)
    parser.add_argument('--path', type=str, default=None)
    parser.add_argument('--objective', type=str, required=True)
    parser.add_argument('--lamda', type=float, default=None)
    parser.add_argument('--penalty_factor', type=float, default=1.0)
    # Add the new argument to load warm-up data
    parser.add_argument('--load_initial_data', type=str, default=None, help='Path to pre-computed warm-up data file (.pt).')
    
    args = parser.parse_args()
    kwag = vars(args)
    
    # --- Data Loading Logic ---
    initial_data = None
    if args.load_initial_data and os.path.exists(args.load_initial_data):
        print(f"INFO: [main.py] Found initial data path: {args.load_initial_data}")
        initial_data = torch.load(args.load_initial_data)
    
    # --- Objective Handling ---
    objective_name = kwag.pop('objective')
    
    if objective_name == 'nas_encoder':
        print("INFO: [main.py] Initializing 'nas_encoder' objective.")
        # Pass the loaded data (or None) to the EncoderBO constructor
        kwag['objective'] = EncoderBO(initial_data=initial_data)
        MerCBO(**kwag)
        sys.exit(0)
    
    # Keep other objectives as they were
    if objective_name == 'transformer':
        print("INFO: [main.py] Initializing 'transformer' objective.")
        kwag['objective'] = TransformerPerplexity()
        MerCBO(**kwag)
        sys.exit(0)
    
    if 'random_seed_config' in kwag:
        # Logic for ising/labs...
        pass

import os
import numpy as np
import torch
import torch.nn as nn
import logging
import yaml
import ipdb
from pathlib import Path
from datetime import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt

def load_config_file(path_to_config, config_file_name):
    config_path = Path(os.path.join(path_to_config, config_file_name))
    config = yaml.safe_load(config_path.read_text())
    return config

def set_N_threads_(N_threads=1):
    logging.info(f'N_threads: {N_threads}')
    os.environ["OMP_NUM_THREADS"] = str(N_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(N_threads)
    os.environ["MKL_NUM_THREADS"] = str(N_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(N_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(N_threads)
    return N_threads

def get_N_colors(N, colormap):
    def get_colors(inp, colormap, vmin=None, vmax=None):
        norm = plt.Normalize(vmin, vmax)
        return colormap(norm(inp))
    colors = get_colors(np.linspace(1, N, num=N, endpoint=True), colormap)
    return colors

def load_stored_data_single_model(path_load, model_name, np_name_xx="xx.npy"):

    logging.info('Loading ' + os.path.join(path_load, model_name + '_' + np_name_xx) + '...')
    with open(os.path.join(path_load, model_name + '_' + np_name_xx), 'rb') as ff:
        loaded_xx = np.load(ff)
    
    logging.info('    # loaded_xx.shape = ' + str(loaded_xx.shape))
    logging.info('    # Noise samples = ' + str(loaded_xx.shape[0]))
    logging.info('    # Cosmo samples = ' + str(loaded_xx.shape[1]))
    logging.info('    # Augs samples = ' + str(loaded_xx.shape[2]))
    logging.info('    # xx dimensions = ' + str(loaded_xx.shape[3:]))
    
    return loaded_xx

class DataLoader:
    def __init__(self, xx, yy, IDs=None):
        """
        Initializes the DataLoader with data and index sets.

        Parameters:
        - xx (np.ndarray): Features.
        - yy (np.ndarray): Labels.
        """
        self.xx = xx
        self.yy = yy
        if IDs is not None:
            self.IDs = IDs

        self.NN = self.xx.shape[0]
        self.xx_dim = self.xx.shape[1:]
    
    def __call__(
        self,
        batch_size,
        seed="random",
        custom_idxs=None,
        to_torch=False,
        device="cpu",
        return_IDs=False
    ):
        """
        Generates a batch of data by randomly sampling indices.

        Parameters:
        - batch_size (int): Number of samples in the batch.
        - seed (int or "random"): Random seed for reproducibility.
        - return_idxs_sampled (bool): Whether to return sampled indices.
        - custom_idxs (np.ndarray or None): Specific indices to use instead of random sampling.
        - to_torch (bool): Whether to convert output to PyTorch tensors.
        - device (str): PyTorch device ("cpu" or "cuda").

        Returns:
        - xx_batch (np.ndarray or torch.Tensor): Sampled feature batch.
        - yy_batch (np.ndarray or torch.Tensor): Sampled target batch.
        - batch_idxs (np.ndarray, optional): Sampled indices if return_idxs_sampled=True.

        Example Usage:
        >>> loader = DataLoader(features, targets, idxs_dset)
        >>> xx_batch, yy_batch = loader(batch_size=32, seed=42)
        """

        logging.debug(f"🚀 Generating batch with batch_size={batch_size}, seed={seed}, custom_idxs={'Provided' if custom_idxs is not None else 'None'}")

        # Generate seed if "random" mode is selected
        if seed == "random":
            seed = datetime.now().microsecond % 13037
        np.random.seed(seed)

        # Sample batch indices
        if custom_idxs is None:
            if batch_size > self.NN:
                logging.error("❌ ERROR: batch_size must be smaller than len(idxs_dset).")
                raise ValueError("batch_size must be smaller than len(idxs_dset).")
            batch_idxs = np.random.choice(self.NN, batch_size, replace=False)
        else:
            if len(custom_idxs) > self.NN:
                logging.error("❌ ERROR: len(custom_idxs) must be smaller than len(idxs_dset).")
                raise ValueError("len(custom_idxs) must be smaller than len(idxs_dset).")
            batch_idxs = np.array(custom_idxs)

        batch_size = len(batch_idxs)

        xx_batch = self.xx[batch_idxs]
        yy_batch = self.yy[batch_idxs]
        if return_IDs:
            IDs_batch = self.IDs[batch_idxs]

        # Convert to PyTorch tensors if requested
        if to_torch:
            xx_batch = torch.tensor(xx_batch, dtype=torch.float32, device=device)
            yy_batch = torch.tensor(yy_batch, dtype=torch.long, device=device)
            if return_IDs:
                IDs_batch = torch.tensor(IDs_batch, dtype=torch.long, device=device)

        logging.debug("✅ Batch generation complete.")

        if return_IDs:
            return xx_batch, yy_batch, IDs_batch
        else:
            return xx_batch, yy_batch
        
def load_dset_wrapper(path_load, list_model_names, np_name_xx="xx.npy", split_ratio=None):
    """
    Loads dataset from stored models and optionally splits it into training and validation sets.

    Parameters:
    - path_load (str): Path to load data from.
    - list_model_names (list of str): List of model names.
    - np_name_xx (str): Filename of the stored NumPy data (default: "xx.npy").
    - split_ratio (float, optional): If provided, splits the dataset into train and validation.
                                      Should be a value between 0 and 1 (e.g., 0.8 for 80% train, 20% validation).

    Returns:
    - DataLoader (train_data) if no split
    - Tuple (train_data, val_data) if split_ratio is provided
    """

    xx, yy, IDs = None, None, None  # Initialize as None to handle first iteration properly

    for ii, model_name in enumerate(list_model_names):
        loaded_xx = load_stored_data_single_model(path_load, model_name, np_name_xx=np_name_xx)
        flatten_xx = np.reshape(loaded_xx, (np.prod(loaded_xx.shape[0:3]), loaded_xx.shape[-1]))
        tmp_yy = np.array([ii] * flatten_xx.shape[0], dtype=np.int64)
        tmp_IDs = np.arange(flatten_xx.shape[0])

        if xx is None:
            xx, yy, IDs = flatten_xx, tmp_yy, tmp_IDs  # Direct assignment for the first batch
        else:
            xx = np.concatenate((xx, flatten_xx), axis=0)
            yy = np.concatenate((yy, tmp_yy), axis=0)
            IDs = np.concatenate((IDs, tmp_IDs), axis=0)

    yy = yy.astype(np.int64)
    IDs = IDs.astype(np.int64)

    # Shuffle the data before splitting
    indices = np.random.permutation(xx.shape[0])
    xx, yy, IDs = xx[indices], yy[indices], IDs[indices]

    # If split_ratio is provided, split into train and validation sets
    if split_ratio:
        split_idx = int(xx.shape[0] * split_ratio)
        xx_train, yy_train, IDs_train = xx[:split_idx], yy[:split_idx], IDs[:split_idx]
        xx_val, yy_val, IDs_val = xx[split_idx:], yy[split_idx:], IDs[split_idx:]

        train_data = DataLoader(xx_train, yy_train, IDs_train)
        val_data = DataLoader(xx_val, yy_val, IDs_val)
        return train_data, val_data

    # Return only one dataset if no split is required
    return DataLoader(xx, yy, IDs)

def create_mlp(input_dim, hidden_layers, output_dim):
    """
    Creates a flexible MLP model.

    Parameters:
    - input_dim (int): Number of input features.
    - hidden_layers (list of int): List containing the number of units for each hidden layer.
    - output_dim (int): Number of output classes (for multi-class classification).

    Returns:
    - nn.Sequential: The MLP model.
    """
    layers = []
    prev_dim = input_dim

    # Add hidden layers
    for hidden_dim in hidden_layers:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(nn.ReLU())  # Activation function
        prev_dim = hidden_dim

    # Add output layer (no activation since we'll use CrossEntropyLoss)
    layers.append(nn.Linear(prev_dim, output_dim))

    return nn.Sequential(*layers)

def train_model(
    dset_train, model, loss_function,
    NN_epochs=300, NN_batches_per_epoch=10, batch_size=16, lr=1e-3, weight_decay=0., clip_grad_norm=None,
    seed_mode="random", # 'random', 'deterministic' or 'overfit'
    seed=0, # only relevant if mode is 'overfit'
    dset_val=None, batch_size_val=16, path_save=None
    ):
    """
    Trains a model based on the specified training mode.

    Parameters:
    - dset_train: Training dataset.
    - model (torch.nn.Module):
    - NN_epochs (int): Number of training epochs.
    - NN_batches_per_epoch (int): Number of batches per epoch.
    - batch_size (int): Batch size.
    - lr (float): Learning rate.
    - weight_decay (float): Weight decay.
    - clip_grad_norm (float, optional): Gradient clipping norm.
    - seed_mode (str): Seed mode. Must be 'random', 'deterministic', or 'overfit'.
    - seed (int): Seed value (relevant if seed_mode is 'overfit').
    - dset_val: Validation dataset.
    - batch_size_val (int): Batch size for validation.
    - path_save (str, optional): Path to save the model.

    Returns:
    - min_val_loss (float): Minimum validation loss.
    """
    
    optimizer = torch.optim.AdamW([*model.parameters()], lr=lr, betas=(0.9, 0.999), eps=1e-8, amsgrad=False, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=20, threshold=0.01, threshold_mode='abs', factor=0.3, min_lr=1e-8, verbose=True)

    assert seed_mode in ["random", "deterministic", "overfit"], " must belong to one of the following categories: 'random', 'deterministic' or 'overfit'"
    
    if next(model.parameters()).is_cuda: device = "cuda"
    else: device = "cpu"
    
    min_val_loss = None
    for tt in range(NN_epochs):
        logging.info(f"\n-------------------------------------\n-------------- Epoch {tt+1} --------------\n-------------------------------------\n")

        if seed_mode == "deterministic": seed0 = NN_batches_per_epoch*tt
        else: seed0=0
        
        # train the model one epoch
        train_single_epoch(
            dset_train=dset_train, optimizer=optimizer, model=model, loss_function=loss_function,
            NN_batches_per_epoch=NN_batches_per_epoch, batch_size=batch_size, clip_grad_norm=clip_grad_norm,
            seed_mode=seed_mode, seed=seed, seed0=seed0,
            device=device, save_aux_fig_name_epoch=str(tt)
        )
        # evaluation of the model after training one epoch
        min_val_loss, train_loss, val_loss = eval_single_epoch(
            dset_train=dset_train, dset_val=dset_val, scheduler=scheduler, model=model, loss_function=loss_function,
            path_save=path_save, min_val_loss=min_val_loss, batch_size=batch_size_val, seed=seed,
            device=device, save_aux_fig_name=str(tt)
        )
        
    return min_val_loss

def train_single_epoch(
    dset_train, optimizer, model, loss_function,
    NN_batches_per_epoch=10, batch_size=16, clip_grad_norm=None,
    seed_mode="random", seed=0, seed0=0,
    device="cuda", NN_print_progress=None, save_aux_fig_name_epoch=None
    ):
    """
    Trains a model for a single epoch.

    Parameters:
    - dset_train: Training dataset.
    - optimizer: Optimizer for training.
    - model (torch.nn.Module): model
    - NN_batches_per_epoch (int, optional): Number of batches per epoch.
    - batch_size (int, optional): Batch size.
    - clip_grad_norm (float, optional): Gradient clipping norm.
    - seed_mode (str, optional): Seed mode.
    - seed (int, optional): Seed value.
    - seed0 (int, optional): Initial seed value.
    - device (str, optional): Device to use for training.
    - save_aux_fig_name_epoch (str, optional): Auxiliary figure name for saving.
    """
    if NN_print_progress == None: NN_print_progress=10
    if NN_print_progress > NN_batches_per_epoch: NN_print_progress = NN_batches_per_epoch
    
    model.train()
    
    LOSS = {}
    for ii_batch in range(NN_batches_per_epoch):
        
        # draw batch from dataset
        if seed_mode == "random": seed = datetime.now().microsecond %13037
        if seed_mode == "deterministic": seed = seed0 + ii_batch
        xx, yy = dset_train(batch_size, seed=seed, to_torch=True, device=device)
        
        logits = model(xx)
        # ipdb.set_trace()  # Add this line to set an ipdb breakpoint
        LOSS["loss"] = loss_function(logits, yy)  # Compute loss
        
        # perform backpropagation (update weights of the model)
        optimizer.zero_grad()
        LOSS['loss'].backward()
        if isinstance(clip_grad_norm, float):            
            torch.nn.utils.clip_grad_norm_([*model.parameters()], max_norm=clip_grad_norm)
            
        optimizer.step()       
        
        # print progress
        if (ii_batch+1) % int(NN_batches_per_epoch/NN_print_progress) == 0:
            logging.info(f"loss: {LOSS['loss'].item():>7f} | batch: [{ii_batch+1:>5d}/{NN_batches_per_epoch:>5d}]")
        
        if device == "cuda":
            torch.cuda.empty_cache()
        
    return logging.info(f"\n---------- done train epoch ---------")

def eval_single_epoch(
    dset_train, dset_val, scheduler, model, loss_function,
    path_save=None, min_val_loss=None, batch_size=None,
    device="cuda", seed=0, save_aux_fig_name=None
    ):
    """
    Evaluates a model after training for one epoch.

    Parameters:
    - dset_train: Training dataset.
    - dset_val: Validation dataset.
    - scheduler: Learning rate scheduler.
    - model (torch.nn.Module):  model.
    - path_save (str, optional): Path to save the model.
    - min_val_loss (float, optional): Minimum validation loss.
    - batch_size (int, optional): Batch size for validation.
    - device (str, optional): Device to use for evaluation.
    - seed (int, optional): Seed value.
    - save_aux_fig_name (str, optional): Auxiliary figure name for saving.

    Returns:
    - min_val_loss (float): Updated minimum validation loss.
    - train_loss (float): Training loss.
    - val_loss (float): Validation loss.
    """
    if batch_size == None:
        batch_size=dset_val.xx.shape[0]

    train_loss = eval_dataset(
        dset_train, batch_size, model=model, loss_function=loss_function, seed=seed, device=device, 
        save_aux_fig_name=None, # <-- replace by save_aux_fig_name if you want to print validation plots
    )
    
    val_loss = eval_dataset(
        dset_val, batch_size, model=model, loss_function=loss_function, seed=seed, device=device,
        save_aux_fig_name=None, # <-- replace by save_aux_fig_name if you want to print validation plots
    )
    
    if min_val_loss == None:
        min_val_loss = val_loss['loss']
        if path_save!=None:
            if not os.path.exists(path_save):
                os.makedirs(path_save)
            ff = open(os.path.join(path_save, 'register.txt'), 'w')
            ff.write('%.4e %.4e\n'%(train_loss['loss'], val_loss['loss']))
            ff.close()
            logging.info(f"Saving Model from Epoch-0")
            torch.save(model.state_dict(), os.path.join(path_save, 'model.pt'))
    else:
        if path_save!=None:
            ff = open(os.path.join(path_save, 'register.txt'), 'a')
            ff.write('%.4e %.4e\n'%(train_loss['loss'], val_loss['loss']))
            ff.close()
    
    logging.info(f"min_val_loss = {min_val_loss:>7f}")
    logging.info(f"train_loss = {train_loss['loss'].item():>7f}")
    logging.info(f"val_loss = {val_loss['loss'].item():>7f}")
        
    if val_loss['loss'] < min_val_loss:
        min_val_loss = val_loss['loss']
        logging.info(f"Saving Model"+path_save)
        if path_save!=None:
            torch.save(model.state_dict(), os.path.join(path_save, 'model.pt'))
                
    scheduler.step(val_loss['loss'])
    
    logging.info(f"\n--------- done eval epoch --------")
    
    if device == "cuda":
        torch.cuda.empty_cache()
    
    return min_val_loss, train_loss, val_loss

def eval_dataset(
        dset, batch_size, model, loss_function,
        seed=0, device="cuda", save_aux_fig_name=None
    ):
    """
    Evaluates a dataset and returns the computed loss.

    Parameters:
    - dset: Dataset to evaluate.
    - batch_size (int): Batch size for evaluation.
    - model (torch.nn.Module): model.
    - seed (int, optional): Seed value.
    - device (str, optional): Device to use for evaluation.
    - save_aux_fig_name (str, optional): Auxiliary figure name for saving.
    - **kwargs: Additional keyword arguments for compute_loss.

    Returns:
    - LOSS (dict): Dictionary containing the output from compute_loss.
    """
    # draw batch from dataset
    xx, yy = dset(batch_size, seed=seed, to_torch=True, device=device)
    
    # obtain model predictions & compute loss
    model.eval()

    LOSS = {}
    with torch.no_grad():
        logits = model(xx)
        LOSS["loss"] = loss_function(logits, yy)  # Compute loss
    
    return LOSS

def classification_predictions_metrics_vs_thresholds(
    true,
    pred,
    min_pred_threshold=0.001,
    max_pred_threshold=0.999,
    NN_pred_threshold = 100
):
    
    list_pred_thresholds=np.linspace(min_pred_threshold, max_pred_threshold, NN_pred_threshold)

    ground_truth = true.flatten().astype(bool)
    P = np.sum(ground_truth)
    N = np.sum(~ground_truth)

    tmp_pred_probabilistic_predictions = pred.flatten()
    
    pred_metrics = {
        'frac_collapsed' : np.zeros(NN_pred_threshold),
        'TPR'            : np.zeros(NN_pred_threshold),
        'TNR'            : np.zeros(NN_pred_threshold),
        'PPV'            : np.zeros(NN_pred_threshold),
        'ACC'            : np.zeros(NN_pred_threshold),
        'F1'             : np.zeros(NN_pred_threshold)
    }
    
    for ii in range(NN_pred_threshold):

        tmp_threshold = list_pred_thresholds[ii]
        tmp_pred_predictions = tmp_pred_probabilistic_predictions > tmp_threshold
        
        pred_metrics['frac_collapsed'][ii] = np.sum(tmp_pred_predictions) / len(tmp_pred_predictions)
        
        TP = np.sum(tmp_pred_predictions * ground_truth)
        TN = np.sum(~tmp_pred_predictions * ~ground_truth)
        FP = np.sum(tmp_pred_predictions * ~ground_truth)
        FN = np.sum(~tmp_pred_predictions * ground_truth)

        pred_metrics['TPR'][ii] = TP / P
        pred_metrics['TNR'][ii] = TN / N
        pred_metrics['PPV'][ii] = TP / (TP + FP)
        pred_metrics['ACC'][ii] = (TP + TN) / (P + N)
        pred_metrics['F1'][ii] = 2*TP / (2*TP + FP + FN)
        
    return list_pred_thresholds, pred_metrics

def plot_classification_metrics_vs_classification_threshold(
    list_classification_thresholds,
    classification_metrics
):
    """
    Plots classification metrics (TPR, TNR, PPV, ACC, F1) vs. classification thresholds.

    Parameters:
    - list_classification_thresholds (array-like): List of classification threshold values.
    - classification_metrics (dict): Dictionary containing metric arrays for different thresholds.

    Returns:
    - fig (matplotlib.figure.Figure): The plotted figure.
    """

    # 1️⃣ Create the Figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # 2️⃣ Plot classification Metrics vs. Threshold
    ax.plot(list_classification_thresholds, classification_metrics['TPR'], c='limegreen', lw=2, label="TPR")
    ax.plot(list_classification_thresholds, classification_metrics['TNR'], c='royalblue', lw=2, label="TNR")
    ax.plot(list_classification_thresholds, classification_metrics['PPV'], c='purple', lw=2, label="PPV")
    ax.plot(list_classification_thresholds, classification_metrics['ACC'], c='orange', lw=2, label="ACC")
    ax.plot(list_classification_thresholds, classification_metrics['F1'], c='gold', lw=2, label="F1")

    # 3️⃣ Formatting
    ax.set_xlabel(r'Classification Threshold', size=20)
    ax.set_ylabel(r'Score', size=20)
    ax.tick_params(axis='both', length=5, width=2, labelsize=14)

    # Fix the limits of both axes between 0 and 1
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)

    # Improve legend
    ax.legend(loc='upper right', fontsize=13, fancybox=True, shadow=True, ncol=5)

    # Make spines (borders) look better
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    plt.tight_layout()
    
    return fig


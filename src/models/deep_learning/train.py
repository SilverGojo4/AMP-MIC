# pylint: disable=line-too-long, import-error, wrong-import-position, too-many-locals, too-many-statements, invalid-name, too-many-arguments, too-many-positional-arguments, too-many-branches, too-many-nested-blocks
"""
Deep Learning Training Module

This module provides functions for training deep learning models in the AMP-MIC project.
It includes the main training pipeline and model training logic with Grid Search, validation,
and early stopping.
"""
# ============================== Standard Library Imports ==============================
import logging
import os
import sys
import time
from itertools import product
from random import sample
from typing import Optional

# ============================== Third-Party Library Imports ==============================
import matplotlib.pyplot as plt
import torch
from matplotlib import style
from torch import nn
from torch.amp import GradScaler, autocast  # type: ignore
from torch.optim import SGD, Adam, AdamW  # type: ignore
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# ============================== Project Root Path Setup ==============================
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
LOGGING_PATH = os.path.join(BASE_PATH, "src/utils/logging_toolkit/src/python")

# Fix Python Path
if BASE_PATH not in sys.path:
    sys.path.append(BASE_PATH)
if LOGGING_PATH not in sys.path:
    sys.path.append(LOGGING_PATH)

# ============================== Project-Specific Imports ==============================
# Logging configuration and custom logger
from setup_logging import CustomLogger

# Deep Learning Utility Functions
from src.models.deep_learning.ania import ANIA
from src.models.deep_learning.fcgr_ania import FCGRANIA
from src.models.deep_learning.utils import (
    extract_cgr_features_and_target_for_dl,
    extract_word_embedding_features_for_dl,
    get_hyperparameter_settings,
    read_json_config,
)
from src.models.deep_learning.word_embedding_ania import WordEmbeddingANIA


# ============================== Custom Function ==============================
def plot_loss_curve(
    train_losses: list[float],
    val_losses: list[float],
    output_path: str,
) -> None:
    """
    Plot training and validation loss curves and save the figure.

    Parameters
    ----------
    train_losses : list of float
        Training loss for each epoch.
    val_losses : list of float
        Validation loss for each epoch.
    output_path : str
        Path to save the plotted figure (including filename, without extension).
    logger : CustomLogger, optional
        Logger for reporting status (default: None).
    """
    style.use("ggplot")
    plt.figure(figsize=(10, 6))
    plt.plot(
        range(1, len(train_losses) + 1),
        train_losses,
        label="Train Loss",
        color="#82B0D2",
        linewidth=2,
    )  # 藍色
    plt.plot(
        range(1, len(val_losses) + 1),
        val_losses,
        label="Validation Loss",
        color="#FFBE7A",
        linewidth=2,
    )
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def train_model(
    X_train_fcgr: torch.Tensor,  # FCGR input for ANIA
    X_train_word: Optional[
        torch.Tensor
    ],  # Word embedding input for ANIA (None for single-input models)
    y_train: torch.Tensor,
    model_type: str,
    hyperparams_config_path: str,
    model_output_path: str,
    logger: CustomLogger,
    train_split: float = 0.8,
    patience: int = 10,
    device: str = "cuda:0",
    random_search: bool = False,
    num_random_samples: int = 50,
) -> Optional[nn.Module]:
    """
    Train a deep learning model with Grid Search, validation, and early stopping, and save it.

    Parameters
    ----------
    X_train_fcgr : torch.Tensor
        FCGR training feature set of shape (batch_size, in_channels, height, width) for 'fcgr_ania' or 'ania'.
        For 'word_embedding_ania', this is the word embedding feature set if X_train_word is None.
    X_train_word : Optional[torch.Tensor]
        Word embedding training feature set of shape (batch_size, in_channels, max_len) for 'word_embedding_ania' or 'ania'.
        None for single-input models like 'fcgr_ania'.
    y_train : torch.Tensor
        Training target variable of shape (batch_size, 1).
    model_type : str
        Type of model to train ('fcgr_ania', 'word_embedding_ania', 'ania').
    hyperparams_config_path : str
        Path to the JSON file containing hyperparameter ranges.
    model_output_path : str
        Path to save the trained model.
    logger : CustomLogger
        Logger instance for tracking progress and errors.
    train_split : float, optional
        Proportion of data to use for training (default is 0.8).
    patience : int, optional
        Number of epochs to wait for improvement in validation loss before early stopping (default is 10).
    device : str, optional
        Device to run the model on (e.g., 'cuda:0', default is 'cuda:0').
    random_search : bool, optional
        Whether to perform random search instead of full grid search (default is False).
    num_random_samples : int, optional
        Number of random hyperparameter combinations to sample if random_search is True (default is 50).

    Returns
    -------
    Optional[nn.Module]
        Trained model instance, or None if training fails.
    """
    try:
        # Log training start
        logger.info(msg=f"/ Task: Train '{model_type}' model")
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

        # Enable cuDNN optimization for faster convolution operations
        if "cuda" in device and torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            logger.log_with_borders(
                level=logging.INFO,
                message="'cuDNN' benchmark mode enabled for optimized convolution performance\n"
                f"'cuDNN' enabled: {torch.backends.cudnn.enabled}, 'Benchmark': {torch.backends.cudnn.benchmark}",
                border="|",
                length=100,
            )
            logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

        # Configure the device (GPU or CPU) and log GPU information if applicable
        gpu_id = None
        if "cuda" in device:
            num_gpus = torch.cuda.device_count()
            device_idx = int(device.split(":")[1]) if ":" in device else 0
            if device_idx >= num_gpus:
                device = "cuda:0"
            if not torch.cuda.is_available():
                logger.log_with_borders(
                    level=logging.WARNING,
                    message="CUDA is not available, falling back to CPU.",
                    border="|",
                    length=100,
                )
                device = "cpu"
            else:
                torch.cuda.set_device(device)
                gpu_id = torch.cuda.current_device()
                if gpu_id != device_idx:
                    logger.warning(
                        f"Current GPU ID ({gpu_id}) does not match specified device index ({device_idx})."
                    )
                torch.cuda.reset_max_memory_allocated(gpu_id)
                total_memory = torch.cuda.get_device_properties(gpu_id).total_memory / (
                    1024**3
                )
                allocated_memory = torch.cuda.memory_allocated(gpu_id) / (1024**3)
                logger.log_with_borders(
                    level=logging.INFO,
                    message=f"Using GPU device.\n"
                    f"  - Specified device: '{device}'\n"
                    f"  - Active GPU ID: {gpu_id}\n"
                    f"  - Total Memory: {total_memory:.2f} GB\n"
                    f"  - Initial Allocated Memory: {allocated_memory:.2f} GB",
                    border="|",
                    length=100,
                )
            logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

        # Record start time for the entire training process
        start_time = time.time()

        # Load hyperparameters
        config = read_json_config(hyperparams_config_path, logger)
        param_grid = get_hyperparameter_settings(config, model_type, logger)
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

        # Split data into training and validation sets
        num_samples = y_train.size(0)
        train_size = int(train_split * num_samples)
        val_size = num_samples - train_size

        if model_type == "ania":
            if X_train_word is None:
                raise ValueError("X_train_word must be provided for 'ania' model.")
            X_train_fcgr_split, X_val_fcgr = torch.split(
                X_train_fcgr, [train_size, val_size]
            )
            X_train_word_split, X_val_word = torch.split(
                X_train_word, [train_size, val_size]
            )
            y_train_split, y_val = torch.split(y_train, [train_size, val_size])
            # Move data to GPU
            if "cuda" in device and torch.cuda.is_available():
                X_train_fcgr_split, X_val_fcgr = X_train_fcgr_split.to(
                    device
                ), X_val_fcgr.to(device)
                X_train_word_split, X_val_word = X_train_word_split.to(
                    device
                ), X_val_word.to(device)
                y_train_split, y_val = y_train_split.to(device), y_val.to(device)
            train_dataset = TensorDataset(
                X_train_fcgr_split, X_train_word_split, y_train_split
            )
            val_dataset = TensorDataset(X_val_fcgr, X_val_word, y_val)
        else:
            X_train_split, X_val = torch.split(X_train_fcgr, [train_size, val_size])
            y_train_split, y_val = torch.split(y_train, [train_size, val_size])
            # Move data to GPU
            if "cuda" in device and torch.cuda.is_available():
                X_train_split = X_train_split.to(device)
                y_train_split = y_train_split.to(device)
                X_val = X_val.to(device)
                y_val = y_val.to(device)
            train_dataset = TensorDataset(X_train_split, y_train_split)
            val_dataset = TensorDataset(X_val, y_val)

        logger.log_with_borders(
            level=logging.INFO,
            message=f"Data split:\n"
            f"  • Training samples: {train_size}\n"
            f"  • Validation samples: {val_size}",
            border="|",
            length=100,
        )
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

        # Move data to GPU once before creating DataLoader (if using GPU)
        if "cuda" in device and torch.cuda.is_available():
            logger.log_with_borders(
                level=logging.INFO,
                message=f"Moved training and validation data to '{device}'",
                border="|",
                length=100,
            )
            logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

        # Calculate total number of hyperparameter combinations
        param_combinations = list(product(*param_grid.values()))
        total_all_combinations = len(param_combinations)
        total_hyperparameters = len(param_grid)

        # Apply Random Search: sample a fixed number of combinations randomly
        if random_search:
            sample_size = min(num_random_samples, total_all_combinations)
            param_combinations = sample(param_combinations, sample_size)
        else:
            sample_size = "N/A"
        logger.log_with_borders(
            level=logging.INFO,
            message=f"Performing {'RandomSearch' if random_search else 'GridSearch'} for '{model_type}':\n"
            f"  • Total number of hyperparameters: {total_hyperparameters}\n"
            f"  • Number of hyperparameter combinations: {total_all_combinations}\n"
            f"  • Sampled combinations: {sample_size}",
            border="|",
            length=100,
        )
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

        # Initialize variables to track the best model and its performance
        best_val_loss = float("inf")
        best_model = None
        best_params = None
        best_combination_idx = -1
        best_features = None
        best_gradients = None
        best_attn_weights = None
        all_train_losses = []
        all_val_losses = []

        # Initialize GradScaler for mixed precision training with 'cuda' device
        scaler = GradScaler("cuda")

        # Perform grid search over hyperparameter combinations with a progress bar
        for idx, params in tqdm(
            enumerate(param_combinations, 1),
            total=len(param_combinations),
            desc=(
                "Grid Search Progress"
                if not random_search
                else "Random Search Progress"
            ),
        ):
            param_dict = dict(zip(param_grid.keys(), params))

            # Create DataLoader with the current batch size from param_dict
            train_loader = DataLoader(
                train_dataset, batch_size=param_dict["batch_size"], shuffle=True
            )
            val_loader = DataLoader(
                val_dataset, batch_size=param_dict["batch_size"], shuffle=False
            )

            # Initialize the model based on model_type
            if model_type == "fcgr_ania":
                model = FCGRANIA(
                    in_channels=1,
                    inception_out_channels=param_dict["inception_out_channels"],
                    num_heads=param_dict["num_heads"],
                    d_model=param_dict["d_model"],
                    dense_hidden_dim=param_dict["dense_hidden_dim"],
                    dropout_rate=param_dict["dropout_rate"],
                ).to(device)
            elif model_type == "word_embedding_ania":
                model = WordEmbeddingANIA(
                    in_channels=(
                        X_train_fcgr.shape[1]
                        if X_train_word is None
                        else X_train_word.shape[1]
                    ),
                    inception_out_channels=param_dict["inception_out_channels"],
                    num_heads=param_dict["num_heads"],
                    d_model=param_dict["d_model"],
                    dense_hidden_dim=param_dict["dense_hidden_dim"],
                    dropout_rate=param_dict["dropout_rate"],
                ).to(device)
            elif model_type == "ania":
                if X_train_word is None:
                    raise ValueError("X_train_word must be provided for 'ania' model.")
                model = ANIA(
                    fcgr_in_channels=1,
                    word_in_channels=X_train_word.shape[1],
                    fcgr_inception_out_channels=param_dict[
                        "fcgr_inception_out_channels"
                    ],
                    word_inception_out_channels=param_dict[
                        "word_inception_out_channels"
                    ],
                    num_heads=param_dict["num_heads"],
                    d_model=param_dict["d_model"],
                    dense_hidden_dim=param_dict["dense_hidden_dim"],
                    dropout_rate=param_dict["dropout_rate"],
                ).to(device)
            else:
                raise ValueError(f"Unsupported model_type: {model_type}")

            # Define the optimizer based on the hyperparameter settings
            optimizer_class = {
                "adam": Adam,
                "adamw": AdamW,
                "sgd": lambda params, lr, weight_decay: SGD(
                    params, lr=lr, momentum=0.9, weight_decay=weight_decay
                ),
            }.get(param_dict["optimizer"], Adam)
            optimizer = optimizer_class(
                model.parameters(),
                lr=param_dict["learning_rate"],
                weight_decay=param_dict["weight_decay"],
            )

            # Define the loss function based on the hyperparameter settings
            criterion_class = {
                "mse": nn.MSELoss,
                "l1": nn.L1Loss,
                "smooth_l1": nn.SmoothL1Loss,
            }.get(param_dict["loss_function"], nn.MSELoss)
            criterion = criterion_class()

            # Variables for early stopping within this hyperparameter combination
            best_combination_val_loss = float("inf")
            epochs_no_improve = 0
            combination_train_losses = []
            combination_val_losses = []

            # Training loop for the current hyperparameter combination
            for _ in range(param_dict["epochs"]):
                model.train()
                total_train_loss = 0.0
                for batch_data in train_loader:
                    optimizer.zero_grad()
                    with autocast("cuda"):
                        if model_type == "ania":
                            batch_x_fcgr, batch_x_word, batch_y = batch_data
                            outputs, _ = model(batch_x_fcgr, batch_x_word)
                        else:
                            batch_x, batch_y = batch_data
                            outputs, _ = model(batch_x)
                        train_loss = criterion(outputs, batch_y)
                    scaler.scale(train_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    total_train_loss += train_loss.item()

                # Validation with mixed precision
                model.eval()
                total_val_loss = 0.0
                with torch.no_grad():
                    with autocast("cuda"):
                        for batch_data in val_loader:
                            if model_type == "ania":
                                batch_x_fcgr, batch_x_word, batch_y = batch_data
                                val_outputs, _ = model(batch_x_fcgr, batch_x_word)
                            else:
                                batch_x, batch_y = batch_data
                                val_outputs, _ = model(batch_x)
                            val_loss = criterion(val_outputs, batch_y)
                            total_val_loss += val_loss.item()

                avg_train_loss = total_train_loss / len(train_loader)
                avg_val_loss = total_val_loss / len(val_loader)
                combination_train_losses.append(avg_train_loss)
                combination_val_losses.append(avg_val_loss)

                # Check for early stopping based on validation loss improvement
                if avg_val_loss < best_combination_val_loss:
                    best_combination_val_loss = avg_val_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        break

            # Store losses for this combination
            all_train_losses.append(combination_train_losses)
            all_val_losses.append(combination_val_losses)

            # Evaluate final validation loss for this combination
            if best_combination_val_loss < best_val_loss and not torch.isnan(
                torch.tensor(best_combination_val_loss)
            ):
                best_val_loss = best_combination_val_loss
                best_model = model
                best_params = param_dict
                best_combination_idx = idx - 1
                if model_type == "ania":
                    best_features = {
                        "fcgr": {
                            k: v.clone().detach().cpu()
                            for k, v in model.fcgr_inception.features.items()
                        },
                        "word": {
                            k: v.clone().detach().cpu()
                            for k, v in model.word_inception.features.items()
                        },
                    }
                    best_gradients = {
                        "fcgr": {
                            k: v.clone().detach().cpu()
                            for k, v in model.fcgr_inception.gradients.items()
                        },
                        "word": {
                            k: v.clone().detach().cpu()
                            for k, v in model.word_inception.gradients.items()
                        },
                    }
                else:
                    best_features = {
                        k: v.clone().detach().cpu()
                        for k, v in model.inception.features.items()
                    }
                    best_gradients = {
                        k: v.clone().detach().cpu()
                        for k, v in model.inception.gradients.items()
                    }
                best_attn_weights = (
                    model.attn_weights.clone().detach().cpu()
                    if model.attn_weights is not None
                    else None
                )

        # Check if a best model was found
        if best_model is None:
            logger.log_with_borders(
                level=logging.ERROR,
                message="No valid model was found during grid search. All combinations failed to improve validation loss.",
                border="|",
                length=100,
            )
            raise ValueError("Training failed: No best model identified.")

        # -------------------- Process the best model --------------------
        best_train_losses = all_train_losses[best_combination_idx]
        best_val_losses = all_val_losses[best_combination_idx]
        loss_plot_path = os.path.join(
            os.path.dirname(model_output_path),
            f"{model_type}_loss_curve_best_combination.png",
        )
        plot_loss_curve(
            train_losses=best_train_losses,
            val_losses=best_val_losses,
            output_path=loss_plot_path,
        )

        # -------------------- Generate and save explainability data for the best model --------------------
        # Log the results of the best model and the saved state
        param_str = "\n".join([f"    ▸ '{k}': {v}" for k, v in best_params.items()])  # type: ignore
        logger.log_with_borders(
            level=logging.INFO,
            message=f"Best results for '{model_type}':\n"
            f"  • Validation Loss: {best_val_loss:.4f}\n"
            f"  • Hyperparameters:\n{param_str}",
            border="|",
            length=100,
        )
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

        # Record end time and final GPU memory usage
        total_time = time.time() - start_time
        if "cuda" in device and torch.cuda.is_available() and gpu_id is not None:
            final_allocated_memory = torch.cuda.memory_allocated(gpu_id) / (1024**3)
            final_reserved_memory = torch.cuda.memory_reserved(gpu_id) / (1024**3)
            final_max_memory = torch.cuda.max_memory_allocated(gpu_id) / (1024**3)
            logger.log_with_borders(
                level=logging.INFO,
                message=f"Total training completed for '{model_type}' model:\n"
                f"  • Time: {total_time:.2f}s\n"
                f"  • Final GPU Memory Usage:\n"
                f"    ▸ Allocated: {final_allocated_memory:.2f} GB\n"
                f"    ▸ Reserved: {final_reserved_memory:.2f} GB\n"
                f"    ▸ Max Allocated: {final_max_memory:.2f} GB",
                border="|",
                length=100,
            )
        else:
            logger.log_with_borders(
                level=logging.INFO,
                message=f"Total training completed for '{model_type}' model:\n"
                f"  • Time: {total_time:.2f}s\n"
                f"  • Running on CPU",
                border="|",
                length=100,
            )
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

        # Save the complete model state, including parameters and explainability data
        os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
        torch.save(
            {
                "state_dict": best_model.state_dict(),
                "features": best_features,
                "gradients": best_gradients,
                "attn_weights": best_attn_weights,
                "hyperparams": best_params,
                "input_shape": (
                    tuple(X_train_fcgr.shape[1:])
                    if model_type != "ania"
                    else {
                        "fcgr": tuple(X_train_fcgr.shape[1:]),
                        "word": tuple(X_train_word.shape[1:]),  # type: ignore
                    }
                ),
                "train_stats": {
                    "train_losses": best_train_losses,
                    "val_losses": best_val_losses,
                    "best_epoch": len(best_train_losses),
                },
                "val_metrics": {"best_val_loss": best_val_loss},
                "training_time": total_time,
                "torch_version": torch.__version__,
            },
            model_output_path + "_full.pt",
        )
        logger.log_with_borders(
            level=logging.INFO,
            message=f"Full model state with explainability data saved to\n'{model_output_path}_full.pt'",
            border="|",
            length=100,
        )
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

        return best_model

    except Exception:
        logger.exception(msg=f"Unexpected error in 'train_model()' for '{model_type}'.")
        raise


# ============================== Pipeline Entry Point ==============================
def run_train_dl_pipeline(
    base_path: str,
    logger: CustomLogger,
    model_type=None,
    train_split: float = 0.8,
    patience: int = 10,
    device: str = "cuda:0",
    random_search: bool = False,
    num_random_samples: int = 50,
) -> None:
    """
    Run the deep learning training pipeline for multiple strains.

    Parameters
    ----------
    base_path : str
        Absolute base path of the project (used to construct file paths).
    logger : CustomLogger
        Logger for structured logging throughout the pipeline.
    model_type : str or list, optional
        Specific model type(s) to train ('fcgr_ania', 'word_embedding_ania', 'ania'). If None or 'all', trains all models.
    train_split : float, optional
        Proportion of data to use for training (default is 0.8).
    patience : int, optional
        Number of epochs to wait for improvement in validation loss before early stopping (default is 10).
    device : str, optional
        Device to run the model on (e.g., 'cuda:0', default is 'cuda:0').
    random_search : bool, optional
        Whether to perform random search instead of full grid search (default is False).
    num_random_samples : int, optional
        Number of random hyperparameter combinations to sample if random_search is True (default is 50).

    Returns
    -------
    None
    """
    try:
        # Mapping of full strain names to their corresponding suffixes
        strains = {
            "Escherichia coli": "EC",
            "Pseudomonas aeruginosa": "PA",
            "Staphylococcus aureus": "SA",
        }

        # Define models to train
        all_models = ["fcgr_ania", "word_embedding_ania", "ania"]
        if isinstance(model_type, list):
            models_to_train = model_type if "all" not in model_type else all_models
        else:
            models_to_train = (
                [model_type] if model_type and model_type != "all" else all_models
            )

        # Define the target column and metadata columns
        target_column = "Log MIC Value"
        metadata_columns = ["ID", "Sequence", "Targets"]

        # Path to hyperparameter configuration
        hyperparams_config_path = os.path.join(
            base_path, "configs/deep_learning/hyperparameters.json"
        )

        # Define feature sets for each model type
        feature_sets = {
            "fcgr_ania": [
                {
                    "name": "CGR",
                    "start_idx": 315,
                    "end_idx": 570,
                    "height": 16,
                    "width": 16,
                    "suffix": "_cgr",
                }
            ],
            "word_embedding_ania": [
                {
                    "name": "WordEmbedding",
                    "llm_name": "BERT-BFD",
                    "height": 64,
                    "width": None,
                    "suffix": "_embed_bert_bfd",
                }
            ],
            "ania": [
                {
                    "name": "CGR",
                    "start_idx": 315,
                    "end_idx": 570,
                    "height": 16,
                    "width": 16,
                    "suffix": "_cgr",
                },
                {
                    "name": "WordEmbedding",
                    "llm_name": "BERT-BFD",
                    "height": 64,
                    "width": None,
                    "suffix": "_embed_bert_bfd",
                },
            ],
        }

        # Loop over each strain type
        for strain_index, (strain, suffix) in enumerate(strains.items(), start=1):
            for model_idx, model in enumerate(models_to_train, start=1):
                train_input_csv_file = os.path.join(
                    base_path, f"data/processed/{suffix}/train.csv"
                )
                train_input_npz_file = (
                    os.path.join(
                        base_path,
                        f"data/processed/{suffix}/train_embed_{feature_sets[model][1]['llm_name']}.npz",
                    )
                    if model == "ania"
                    else (
                        os.path.join(
                            base_path,
                            f"data/processed/{suffix}/train_embed_{feature_sets[model][0]['llm_name']}.npz",
                        )
                        if model == "word_embedding_ania"
                        else None
                    )
                )

                if model == "ania":
                    # Log combined feature header for ANIA
                    logger.info(msg=f"/ {strain_index}.{model_idx}")
                    logger.add_divider(
                        level=logging.INFO, length=110, border="+", fill="-"
                    )
                    logger.log_with_borders(
                        level=logging.INFO,
                        message=f"Strain - '{strain}' -> Model - '{model}' -> Features - 'CGR + WordEmbedding'",
                        border="|",
                        length=110,
                    )
                    logger.add_divider(
                        level=logging.INFO, length=110, border="+", fill="-"
                    )
                    logger.add_spacer(level=logging.INFO, lines=1)

                    # Extract FCGR features
                    _, X_train_fcgr, y_train = extract_cgr_features_and_target_for_dl(
                        file_path=train_input_csv_file,
                        metadata_columns=metadata_columns,
                        target_column=target_column,
                        feature_start_idx=feature_sets[model][0]["start_idx"],
                        feature_end_idx=feature_sets[model][0]["end_idx"],
                        height=feature_sets[model][0]["height"],
                        width=feature_sets[model][0]["width"],
                        logger=logger,
                    )
                    logger.add_spacer(level=logging.INFO, lines=1)
                    # Extract Word Embedding features
                    _, X_train_word, _ = extract_word_embedding_features_for_dl(
                        npz_path=train_input_npz_file,  # type: ignore
                        df_path=train_input_csv_file,
                        metadata_columns=metadata_columns,
                        target_column=target_column,
                        height=feature_sets[model][1]["height"],
                        width=feature_sets[model][1]["width"],
                        logger=logger,
                    )
                    logger.add_spacer(level=logging.INFO, lines=1)

                    model_output_path = os.path.join(
                        base_path,
                        f"experiments/models/{suffix}/deep_learning/{model}_combined",
                    )
                    train_model(
                        X_train_fcgr=X_train_fcgr,
                        X_train_word=X_train_word,
                        y_train=y_train,
                        model_type=model,
                        hyperparams_config_path=hyperparams_config_path,
                        model_output_path=model_output_path,
                        logger=logger,
                        train_split=train_split,
                        patience=patience,
                        device=device,
                        random_search=random_search,
                        num_random_samples=num_random_samples,
                    )
                else:
                    for _, feature in enumerate(feature_sets[model], start=1):
                        logger.info(msg=f"/ {strain_index}.{model_idx}")
                        logger.add_divider(
                            level=logging.INFO, length=110, border="+", fill="-"
                        )
                        logger.log_with_borders(
                            level=logging.INFO,
                            message=f"Strain - '{strain}' -> Model - '{model}' -> Feature - '{feature['name']}'",
                            border="|",
                            length=110,
                        )
                        logger.add_divider(
                            level=logging.INFO, length=110, border="+", fill="-"
                        )
                        logger.add_spacer(level=logging.INFO, lines=1)

                        if model == "fcgr_ania":
                            _, X_train_fcgr, y_train = (
                                extract_cgr_features_and_target_for_dl(
                                    file_path=train_input_csv_file,
                                    metadata_columns=metadata_columns,
                                    target_column=target_column,
                                    feature_start_idx=feature["start_idx"],
                                    feature_end_idx=feature["end_idx"],
                                    height=feature["height"],
                                    width=feature["width"],
                                    logger=logger,
                                )
                            )
                            X_train_word = None
                        elif model == "word_embedding_ania":
                            _, X_train_fcgr, y_train = (
                                extract_word_embedding_features_for_dl(
                                    npz_path=train_input_npz_file,  # type: ignore
                                    df_path=train_input_csv_file,
                                    metadata_columns=metadata_columns,
                                    target_column=target_column,
                                    height=feature["height"],
                                    width=feature["width"],
                                    logger=logger,
                                )
                            )
                            X_train_word = None
                        logger.add_spacer(level=logging.INFO, lines=1)

                        model_output_path = os.path.join(
                            base_path,
                            f"experiments/models/{suffix}/deep_learning/{model}{feature['suffix']}",
                        )
                        train_model(
                            X_train_fcgr=X_train_fcgr,
                            X_train_word=X_train_word,
                            y_train=y_train,
                            model_type=model,
                            hyperparams_config_path=hyperparams_config_path,
                            model_output_path=model_output_path,
                            logger=logger,
                            train_split=train_split,
                            patience=patience,
                            device=device,
                            random_search=random_search,
                            num_random_samples=num_random_samples,
                        )

                # Insert a blank line in the log for readability
                logger.add_spacer(level=logging.INFO, lines=1)

    except Exception:
        logger.exception(msg="Unexpected error in 'run_train_dl_pipeline()'.")
        raise

# pylint: disable=too-many-arguments, line-too-long, import-error, invalid-name, too-many-locals, too-many-branches, wrong-import-position, too-many-positional-arguments, too-many-statements, too-many-nested-blocks
"""
Fine-tuning Module for Deep Learning Models

This module provides a pipeline for fine-tuning pretrained deep learning models in the AMP-MIC project.
It loads a previously trained model, freezes selected layers, and performs additional training on the remaining parameters.
Supports FCGRANIA (2D CGR features), WordEmbeddingANIA (1D word embedding features), and ANIA (combined features).
"""

# ============================== Standard Library Imports ==============================
import logging
import os
import sys
import time
from typing import Optional

# ============================== Third-Party Library Imports ==============================
import torch
from torch import nn
from torch.amp import GradScaler, autocast  # type: ignore
from torch.optim import SGD, Adam, AdamW  # type: ignore
from torch.utils.data import DataLoader, TensorDataset

# ============================== Project Root Path Setup ==============================
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
LOGGING_PATH = os.path.join(BASE_PATH, "src/utils/logging_toolkit/src/python")

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
from src.models.deep_learning.train import plot_loss_curve
from src.models.deep_learning.utils import (
    extract_cgr_features_and_target_for_dl,
    extract_word_embedding_features_for_dl,
)
from src.models.deep_learning.word_embedding_ania import WordEmbeddingANIA


def fine_tune_model(
    base_model_path: str,
    X_train_fcgr: torch.Tensor,
    X_train_word: Optional[torch.Tensor],
    y_train: torch.Tensor,
    model_type: str,
    output_path: str,
    logger: CustomLogger,
    device: str = "cuda:0",
    epochs: int = 10,
    freeze_inception: bool = True,
    batch_size: int = 128,
) -> Optional[nn.Module]:
    """
    Fine-tune a pretrained deep learning model with specified parameters.

    Parameters
    ----------
    base_model_path : str
        Path to the pretrained model checkpoint.
    X_train_fcgr : torch.Tensor
        FCGR training feature set of shape (batch_size, in_channels, height, width) for 'fcgr_ania' or 'ania'.
        For 'word_embedding_ania', this is the word embedding feature set if X_train_word is None.
    X_train_word : Optional[torch.Tensor]
        Word embedding training feature set of shape (batch_size, in_channels, max_len) for 'word_embedding_ania' or 'ania'.
        None for single-input models like 'fcgr_ania'.
    y_train : torch.Tensor
        Training target tensor of shape (batch_size, 1).
    model_type : str
        Type of model to fine-tune ('fcgr_ania', 'word_embedding_ania', or 'ania').
    output_path : str
        Path to save the fine-tuned model.
    logger : CustomLogger
        Logger instance for tracking progress and errors.
    device : str, optional
        Device to run training on (default: 'cuda:0').
    epochs : int, optional
        Number of fine-tuning epochs (default: 10).
    freeze_inception : bool, optional
        Whether to freeze the Inception module (default: True).
    batch_size : int, optional
        Batch size for training (default: 128).

    Returns
    -------
    Optional[nn.Module]
        Fine-tuned model instance, or None if fine-tuning fails.
    """
    try:
        logger.info(
            msg=f"/ Task: Starting 'Fine-Tuning' phase for '{model_type}' model"
        )
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

        # Loaded model
        logger.log_with_borders(
            level=logging.INFO,
            message=f"Successfully loaded trained model from checkpoint:\n'{base_model_path}'",
            border="|",
            length=100,
        )
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")
        checkpoint = torch.load(base_model_path, map_location=device)
        best_hparams = checkpoint["hyperparams"]
        input_shape = checkpoint["input_shape"]
        loss_name = best_hparams.get("loss_function", "mse")

        # Rebuild model based on model_type
        if model_type == "fcgr_ania":
            model = FCGRANIA(
                in_channels=1,
                inception_out_channels=best_hparams["inception_out_channels"],
                d_model=best_hparams["d_model"],
                num_heads=best_hparams["num_heads"],
                dense_hidden_dim=best_hparams["dense_hidden_dim"],
                dropout_rate=best_hparams["dropout_rate"],
            ).to(device)
        elif model_type == "word_embedding_ania":
            model = WordEmbeddingANIA(
                in_channels=input_shape[0],  # embedding_dim from checkpoint
                inception_out_channels=best_hparams["inception_out_channels"],
                d_model=best_hparams["d_model"],
                num_heads=best_hparams["num_heads"],
                dense_hidden_dim=best_hparams["dense_hidden_dim"],
                dropout_rate=best_hparams["dropout_rate"],
            ).to(device)
        elif model_type == "ania":
            if X_train_word is None:
                raise ValueError("X_train_word must be provided for 'ania' model.")
            model = ANIA(
                fcgr_in_channels=1,
                word_in_channels=input_shape["word"][0],  # 從字典中提取
                fcgr_inception_out_channels=best_hparams["fcgr_inception_out_channels"],
                word_inception_out_channels=best_hparams["word_inception_out_channels"],
                d_model=best_hparams["d_model"],
                num_heads=best_hparams["num_heads"],
                dense_hidden_dim=best_hparams["dense_hidden_dim"],
                dropout_rate=best_hparams["dropout_rate"],
            ).to(device)
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        # Load pretrained weights
        model.load_state_dict(checkpoint["state_dict"])

        # Freeze Inception if required
        if freeze_inception:
            if model_type == "ania":
                for param in model.fcgr_inception.parameters():
                    param.requires_grad = False
                for param in model.word_inception.parameters():
                    param.requires_grad = False
            else:
                for param in model.inception.parameters():
                    param.requires_grad = False
            logger.log_with_borders(
                level=logging.INFO,
                message="Froze parameters of the 'Inception module(s)'.",
                border="|",
                length=100,
            )
            logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

        # Setup optimizer
        optimizer_cls = {
            "adam": Adam,
            "adamw": AdamW,
            "sgd": lambda params, lr, weight_decay: SGD(
                params, lr=lr, weight_decay=weight_decay, momentum=0.9
            ),
        }.get(best_hparams["optimizer"], Adam)
        optimizer = optimizer_cls(
            model.parameters(),
            lr=best_hparams["learning_rate"],
            weight_decay=best_hparams["weight_decay"],
        )

        # Setup loss function
        loss_fn = {
            "mse": nn.MSELoss(),
            "l1": nn.L1Loss(),
            "smooth_l1": nn.SmoothL1Loss(),
        }.get(loss_name, nn.MSELoss())

        # Setup DataLoader
        if model_type == "ania":
            dataset = TensorDataset(
                X_train_fcgr.to(device), X_train_word.to(device), y_train.to(device)  # type: ignore
            )
        else:
            dataset = TensorDataset(X_train_fcgr.to(device), y_train.to(device))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        scaler = GradScaler("cuda")
        train_losses = []
        start_time = time.time()

        # Fine-tuning loop
        for _ in range(epochs):
            model.train()
            epoch_loss = 0.0
            for batch_data in dataloader:
                optimizer.zero_grad()
                with autocast("cuda"):
                    if model_type == "ania":
                        batch_x_fcgr, batch_x_word, batch_y = batch_data
                        outputs, _ = model(batch_x_fcgr, batch_x_word)
                    else:
                        batch_x, batch_y = batch_data
                        outputs, _ = model(batch_x)
                    loss = loss_fn(outputs, batch_y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(dataloader)
            train_losses.append(avg_loss)

        # Plot fine-tuning loss curve
        loss_plot_path = os.path.join(
            os.path.dirname(output_path),
            f"{model_type}_loss_curve_finetune.png",
        )
        plot_loss_curve(
            train_losses=train_losses, val_losses=[], output_path=loss_plot_path
        )

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

        # Save fine-tuned model with explainability data
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        save_dict = {
            "state_dict": model.state_dict(),
            "hyperparams": best_hparams,
            "input_shape": input_shape,
            "fine_tune_stats": {
                "train_losses": train_losses,
                "epochs": epochs,
                "fine_tune_time": total_time,
            },
            "torch_version": torch.__version__,
        }
        if model_type == "ania":
            save_dict["features"] = {
                "fcgr": model.fcgr_inception.features,
                "word": model.word_inception.features,
            }
            save_dict["gradients"] = {
                "fcgr": model.fcgr_inception.gradients,
                "word": model.word_inception.gradients,
            }
        else:
            save_dict["features"] = model.inception.features
            save_dict["gradients"] = model.inception.gradients
        save_dict["attn_weights"] = model.attn_weights
        torch.save(save_dict, output_path + "_finetuned_full.pt")
        logger.log_with_borders(
            level=logging.INFO,
            message=f"Saved fine-tuned model to\n'{output_path}_finetuned_full.pt'",
            border="|",
            length=100,
        )
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

        return model

    except Exception:
        logger.exception(
            msg=f"Unexpected error in 'fine_tune_model()' for '{model_type}'."
        )
        raise


def run_fine_tune_pipeline(
    base_path: str,
    logger: CustomLogger,
    model_type=None,
    ft_epochs: int = 10,
    device: str = "cuda:0",
    freeze_inception: bool = True,
) -> None:
    """
    Run the fine-tuning pipeline for pretrained deep learning models.

    Parameters
    ----------
    base_path : str
        Project root path.
    logger : CustomLogger
        Logging object.
    model_type : str, optional
        Type of model to fine-tune ('fcgr_ania' or 'word_embedding_ania').
    ft_epochs : int, optional
        Number of fine-tuning epochs (default: 10).
    device : str, optional
        Device to run training on (default: 'cuda:0').
    freeze_inception : bool, optional
        Whether to freeze the Inception module (default: True).

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

        # Define models to fine-tune
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

        # Loop over each strain type
        for strain_index, (strain, suffix) in enumerate(strains.items(), start=1):
            for model_idx, model in enumerate(models_to_train, start=1):
                train_csv = os.path.join(
                    base_path, f"data/processed/{suffix}/train.csv"
                )
                train_npz = (
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
                        file_path=train_csv,
                        metadata_columns=metadata_columns,
                        target_column=target_column,
                        feature_start_idx=feature_sets[model][0]["start_idx"],
                        feature_end_idx=feature_sets[model][0]["end_idx"],
                        height=feature_sets[model][0]["height"],
                        width=feature_sets[model][0]["width"],
                        logger=logger,
                    )
                    # Extract Word Embedding features
                    _, X_train_word, _ = extract_word_embedding_features_for_dl(
                        npz_path=train_npz,  # type: ignore
                        df_path=train_csv,
                        metadata_columns=metadata_columns,
                        target_column=target_column,
                        height=feature_sets[model][1]["height"],
                        width=feature_sets[model][1]["width"],
                        logger=logger,
                    )
                    logger.add_spacer(level=logging.INFO, lines=1)

                    model_ckpt_path = os.path.join(
                        base_path,
                        f"experiments/models/{suffix}/deep_learning/{model}_combined_full.pt",
                    )
                    output_path = os.path.join(
                        base_path,
                        f"experiments/models/{suffix}/deep_learning/{model}_combined",
                    )

                    # Run fine-tuning for ANIA
                    fine_tune_model(
                        base_model_path=model_ckpt_path,
                        X_train_fcgr=X_train_fcgr,
                        X_train_word=X_train_word,
                        y_train=y_train,
                        model_type=model,
                        output_path=output_path,
                        logger=logger,
                        device=device,
                        epochs=ft_epochs,
                        freeze_inception=freeze_inception,
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
                                    file_path=train_csv,
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
                                    npz_path=train_npz,  # type: ignore
                                    df_path=train_csv,
                                    metadata_columns=metadata_columns,
                                    target_column=target_column,
                                    height=feature["height"],
                                    width=feature["width"],
                                    logger=logger,
                                )
                            )
                            X_train_word = None
                        logger.add_spacer(level=logging.INFO, lines=1)

                        model_ckpt_path = os.path.join(
                            base_path,
                            f"experiments/models/{suffix}/deep_learning/{model}{feature['suffix']}_full.pt",
                        )
                        output_path = os.path.join(
                            base_path,
                            f"experiments/models/{suffix}/deep_learning/{model}{feature['suffix']}",
                        )

                        # Run fine-tuning for single-input models
                        fine_tune_model(
                            base_model_path=model_ckpt_path,
                            X_train_fcgr=X_train_fcgr,
                            X_train_word=X_train_word,
                            y_train=y_train,
                            model_type=model,
                            output_path=output_path,
                            logger=logger,
                            device=device,
                            epochs=ft_epochs,
                            freeze_inception=freeze_inception,
                        )

                # Insert a blank line in the log for readability
                logger.add_spacer(level=logging.INFO, lines=1)

    except Exception:
        logger.exception(msg="Unexpected error in 'run_fine_tune_pipeline()'.")
        raise

# pylint: disable=line-too-long, import-error, wrong-import-position, too-many-locals, too-many-statements, invalid-name, too-many-arguments, too-many-positional-arguments, too-many-branches, too-many-instance-attributes
"""
ANIA model for MIC prediction in the AMP-MIC project.

This module defines the ANIA model, which combines FCGR and word embedding features using separate Inception modules,
flattens their outputs, concatenates them, and processes them through Multi-Head Attention and Dense layers
to predict Minimum Inhibitory Concentration (MIC) values. It includes explainability features such as branch weights,
Grad-CAM, and Attention Scores.
"""
# ============================== Standard Library Imports ==============================
import os
import sys

# ============================== Third-Party Library Imports ==============================
import torch
from torch import nn

# ============================== Project Root Path Setup ==============================
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

# Fix Python Path
if BASE_PATH not in sys.path:
    sys.path.append(BASE_PATH)

# ============================== Project-Specific Imports ==============================
# Deep Learning Utility Functions
from src.models.deep_learning.fcgr_ania import FCGRInceptionModule
from src.models.deep_learning.word_embedding_ania import WordEmbeddingInceptionModule


# ============================== Custom Function ==============================
class ANIA(nn.Module):
    """
    ANIA (An Inception-Attention Network for Predicting the Minimum Inhibitory Concentration)
    for MIC prediction in the AMP-MIC project.

    This model processes FCGR and word embedding features through separate Inception Modules,
    flattens and concatenates their outputs, followed by Multi-Head Attention and Dense layers
    to predict MIC values.
    """

    def __init__(
        self,
        fcgr_in_channels: int = 1,
        word_in_channels: int = 1024,
        fcgr_inception_out_channels: int = 128,
        word_inception_out_channels: int = 128,
        num_heads: int = 8,
        d_model: int = 512,
        dense_hidden_dim: int = 256,
        dropout_rate: float = 0.3,
    ):
        """
        Initialize the ANIA model.

        Parameters
        ----------
        fcgr_in_channels : int
            Number of input channels for FCGR features (default is 1).
        word_in_channels : int
            Number of input channels for word embedding features (default is 1024).
        fcgr_inception_out_channels : int
            Number of output channels from the FCGR Inception Module (default is 128).
        word_inception_out_channels : int
            Number of output channels from the Word Embedding Inception Module (default is 128).
        num_heads : int
            Number of attention heads in Multi-Head Attention (default is 8).
        d_model : int
            Dimension of the model for Multi-Head Attention (default is 512).
        dense_hidden_dim : int
            Hidden dimension of the dense layer (default is 256).
        dropout_rate : float
            Dropout rate for regularization (default is 0.3).
        """
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
            )

        # FCGR Inception Module
        self.fcgr_inception = FCGRInceptionModule(
            in_channels=fcgr_in_channels, out_channels=fcgr_inception_out_channels
        )

        # Word Embedding Inception Module
        self.word_inception = WordEmbeddingInceptionModule(
            in_channels=word_in_channels, out_channels=word_inception_out_channels
        )

        # Flatten dimensions
        self.fcgr_flatten_dim = (
            fcgr_inception_out_channels * 16 * 16
        )  # Assuming 16x16 input
        self.word_flatten_dim = word_inception_out_channels * 64  # Assuming max_len=64

        # Projection to d_model after concatenation
        self.projection = nn.Linear(
            self.fcgr_flatten_dim + self.word_flatten_dim, d_model
        )

        # Multi-Head Attention
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, batch_first=True
        )

        # Dense layers for regression
        self.dense = nn.Sequential(
            nn.Linear(d_model, dense_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(dense_hidden_dim, 1),
        )

        # Initialize weights and attention weights
        self.attn_weights = None
        self._initialize_weights()

    def forward(
        self, fcgr_input: torch.Tensor, word_input: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the ANIA model.

        Parameters
        ----------
        fcgr_input : torch.Tensor
            FCGR input tensor of shape (batch_size, fcgr_in_channels, height, width).
        word_input : torch.Tensor
            Word embedding input tensor of shape (batch_size, word_in_channels, max_len).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Predicted MIC value (batch_size, 1) and attention weights.
        """
        # Process FCGR input through Inception
        fcgr_out = self.fcgr_inception(
            fcgr_input
        )  # (batch_size, fcgr_inception_out_channels, 16, 16)
        fcgr_flat = fcgr_out.view(
            fcgr_out.size(0), -1
        )  # (batch_size, fcgr_flatten_dim)

        # Process Word Embedding input through Inception
        word_out = self.word_inception(
            word_input
        )  # (batch_size, word_inception_out_channels, 64)
        word_flat = word_out.view(
            word_out.size(0), -1
        )  # (batch_size, word_flatten_dim)

        # Concatenate flattened outputs
        combined = torch.cat(
            (fcgr_flat, word_flat), dim=1
        )  # (batch_size, fcgr_flatten_dim + word_flatten_dim)

        # Project to d_model and add sequence dimension
        x = self.projection(combined).unsqueeze(1)  # (batch_size, 1, d_model)

        # Multi-Head Attention
        attn_output, attn_weights = self.attention(
            x, x, x, need_weights=True
        )  # (batch_size, 1, d_model), (batch_size, num_heads, 1, 1)
        if self.training:
            self.attn_weights = attn_weights

        # Remove sequence dimension and process through dense layers
        x = attn_output.squeeze(1)  # (batch_size, d_model)
        x = self.dense(x)  # (batch_size, 1)

        return x, attn_weights

    def _initialize_weights(self) -> None:
        """
        Initialize weights of the model using Xavier uniform initialization.
        """
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Conv1d, nn.Linear)):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def compute_gradcam(
        self, branch_name: str, input_type: str = "fcgr"
    ) -> torch.Tensor:
        """
        Compute Grad-CAM heatmap for a specific branch of either FCGR or Word Embedding Inception.

        Parameters
        ----------
        branch_name : str
            Name of the branch to compute Grad-CAM for ('branch1x1', 'branch3x3', 'branch5x5', 'branch_pool', 'output').
        input_type : str
            Type of input to compute Grad-CAM for ('fcgr' or 'word').

        Returns
        -------
        torch.Tensor
            Grad-CAM heatmap (2D for FCGR, 1D for Word Embedding).
        """
        if input_type == "fcgr":
            inception = self.fcgr_inception
            features = inception.get_features()[
                branch_name
            ]  # (batch_size, channels, height, width)
            gradients = inception.get_gradients()[
                "output"
            ]  # (batch_size, channels, height, width)
            grad_weights = gradients.mean(
                dim=[2, 3], keepdim=True
            )  # (batch_size, channels, 1, 1)
            gradcam = (grad_weights * features).sum(
                dim=1
            )  # (batch_size, height, width)
        elif input_type == "word":
            inception = self.word_inception
            features = inception.get_features()[
                branch_name
            ]  # (batch_size, channels, max_len)
            gradients = inception.get_gradients()[
                "output"
            ]  # (batch_size, channels, max_len)
            grad_weights = gradients.mean(
                dim=2, keepdim=True
            )  # (batch_size, channels, 1)
            gradcam = (grad_weights * features).sum(dim=1)  # (batch_size, max_len)
        else:
            raise ValueError(f"Unsupported input_type: {input_type}")

        if features.shape != gradients.shape:
            raise ValueError(
                f"Features shape {features.shape} does not match gradients shape {gradients.shape}"
            )

        gradcam = torch.relu(gradcam)
        gradcam = gradcam / (gradcam.max() + 1e-8)
        return gradcam[0]  # (height, width) or (max_len) for the first sample

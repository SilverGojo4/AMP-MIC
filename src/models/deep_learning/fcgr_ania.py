# pylint: disable=line-too-long, import-error, wrong-import-position, too-many-locals, too-many-statements, too-many-instance-attributes, too-many-arguments, too-many-positional-arguments
"""
FCGRANIA model for MIC prediction in the AMP-MIC project.

This module defines the FCGRANIA model, which processes FCGR features using an Inception-based architecture,
followed by Multi-Head Attention and Dense layers to predict Minimum Inhibitory Concentration (MIC) values.
It includes explainability features such as branch weights, Grad-CAM, and Attention Scores.
"""
# ============================== Third-Party Library Imports ==============================
import torch
from torch import nn


# ============================== Custom Function ==============================
class BasicConv2d(nn.Module):
    """
    A basic Conv2D block with BatchNorm and ReLU.
    """

    def __init__(self, in_channels: int, out_channels: int, **kwargs) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        self.relu = nn.ReLU(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of BasicConv2d block.
        """
        return self.relu(self.bn(self.conv(x)))


class FCGRInceptionModule(nn.Module):
    """
    Inception Module for FCGR features in the AMP-MIC project.

    This module processes FCGR features using a multi-branch Inception structure,
    extracting multi-scale features through different convolutional paths.
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 128):
        """
        Initialize the FCGR Inception Module.

        Parameters
        ----------
        in_channels : int
            Number of input channels (default is 1 for single-channel FCGR features).
        out_channels : int
            Total number of output channels (default is 128).
        """
        super().__init__()

        if out_channels % 4 != 0:
            raise ValueError(f"out_channels ({out_channels}) must be divisible by 4")

        # Adjust output channels for each branch
        branch_channels = out_channels // 4  # 128 / 4 = 32

        # Branch 1: 1x1 Conv
        self.branch1x1 = BasicConv2d(
            in_channels,
            branch_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
        )

        # Branch 2: 1x1 Conv -> 3x3 Conv
        self.branch3x3_1 = BasicConv2d(
            in_channels,
            branch_channels // 2,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
        )
        self.branch3x3_2 = BasicConv2d(
            branch_channels // 2,
            branch_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )

        # Branch 3: 1x1 Conv -> 3x3 Conv
        self.branch3x3dbl_1 = BasicConv2d(
            in_channels,
            branch_channels // 2,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
        )
        self.branch3x3dbl_2 = BasicConv2d(
            branch_channels // 2,
            branch_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )

        # Branch 4: 1x1 Conv
        self.branch_pool = BasicConv2d(
            in_channels,
            branch_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
        )

        # Learnable weights for each branch
        self.branch_weights = nn.Parameter(torch.ones(4) / 4)

        # Register hooks to save features and gradients
        self.features = {}
        self.gradients = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the BasicConv2d block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Activated output.
        """
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch_pool = self.branch_pool(x)

        # Save features for Grad-CAM
        if self.training:
            self.features["branch1x1"] = branch1x1
            self.features["branch3x3"] = branch3x3
            self.features["branch3x3dbl"] = branch3x3dbl
            self.features["branch_pool"] = branch_pool

        # Apply branch weights
        branch1x1 = branch1x1 * self.branch_weights[0]
        branch3x3 = branch3x3 * self.branch_weights[1]
        branch3x3dbl = branch3x3dbl * self.branch_weights[2]
        branch_pool = branch_pool * self.branch_weights[3]

        out = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        out = torch.cat(out, 1)

        # Save output for Grad-CAM
        if self.training and out.requires_grad:
            out.register_hook(self._save_gradients)
            self.features["output"] = out

        return out

    def _save_gradients(self, grad: torch.Tensor) -> None:
        """
        Hook to save gradients of the output.

        Parameters
        ----------
        grad : torch.Tensor
            Gradient of the output.
        """
        self.gradients["output"] = grad

    def get_branch_weights(self) -> torch.Tensor:
        """
        Get the learned branch weights.

        Returns
        -------
        torch.Tensor
            The learned weights for each branch.
        """
        return self.branch_weights

    def get_features(self) -> dict:
        """
        Get the features of each branch.

        Returns
        -------
        dict
            Dictionary containing the features of each branch.
        """
        return self.features

    def get_gradients(self) -> dict:
        """
        Get the gradients of the output.

        Returns
        -------
        dict
            Dictionary containing the gradients of the output.
        """
        return self.gradients


class FCGRANIA(nn.Module):
    """
    FCGRANIA (FCGR-specific part of An Inception-Attention Network for Predicting the Minimum Inhibitory Concentration)
    for MIC prediction in the AMP-MIC project.

    This model processes FCGR features through an Inception Module, followed by
    Flatten, Multi-Head Attention, and Dense layers to predict MIC values.
    """

    def __init__(
        self,
        in_channels: int = 1,
        inception_out_channels: int = 128,
        num_heads: int = 8,
        d_model: int = 512,
        dense_hidden_dim: int = 256,
        dropout_rate: float = 0.3,
    ):
        """
        Initialize the FCGRANIA model.

        Parameters
        ----------
        in_channels : int
            Number of input channels for FCGR features (default is 1).
        inception_out_channels : int
            Number of output channels from the Inception Module (default is 128).
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

        # Inception Module
        self.inception = FCGRInceptionModule(
            in_channels=in_channels, out_channels=inception_out_channels
        )

        # Projection to d_model for Multi-Head Attention
        self.projection = nn.Linear(inception_out_channels, d_model)

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

        # Initialize weights
        self.attn_weights = None
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the FCGRANIA model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_channels, height, width).

        Returns
        -------
        torch.Tensor
            Predicted MIC value (batch_size, 1).
        """
        x = self.inception(x)  # (batch_size, 128, 16, 16)

        # Reshape to sequence: (batch_size, height*width, channels)
        batch_size, channels, height, width = x.size()
        x = x.view(batch_size, channels, height * width).permute(
            0, 2, 1
        )  # (batch_size, 256, 128)

        # Project to d_model
        x = self.projection(x)  # (batch_size, 256, 512)

        # Multi-Head Attention
        attn_output, attn_weights = self.attention(
            x, x, x, need_weights=True
        )  # (batch_size, 256, 512), (batch_size, num_heads, 256, 256)
        if self.training:
            self.attn_weights = attn_weights

        # Global average pooling over sequence dimension
        x = attn_output.mean(dim=1)  # (batch_size, 512)

        # Dense layers
        x = self.dense(x)  # (batch_size, 1)

        return x, attn_weights

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def compute_gradcam(self, branch_name: str) -> torch.Tensor:
        """
        Compute Grad-CAM heatmap for a specific branch.

        Parameters
        ----------
        branch_name : str
            Name of the branch to compute Grad-CAM for ('branch1x1', 'branch3x3', 'branch3x3dbl', 'branch_pool', 'output').

        Returns
        -------
        torch.Tensor
            Grad-CAM heatmap of shape (height, width), where height and width are determined by the feature map.
        """
        features = self.inception.get_features()[
            branch_name
        ]  # (batch_size, channels, height, width)
        gradients = self.inception.get_gradients()[
            "output"
        ]  # (batch_size, channels, height, width)

        # Check shape consistency
        if features.shape != gradients.shape:
            raise ValueError(
                f"Features shape {features.shape} does not match gradients shape {gradients.shape}"
            )

        # Compute global average of gradients over spatial dimensions
        grad_weights = gradients.mean(
            dim=[2, 3], keepdim=True
        )  # (batch_size, channels, 1, 1)

        # Compute Grad-CAM: weighted sum of feature maps
        gradcam = (grad_weights * features).sum(dim=1)  # (batch_size, height, width)

        # Apply ReLU to keep positive contributions
        gradcam = torch.relu(gradcam)

        # Normalize to [0, 1]
        gradcam = gradcam / (gradcam.max() + 1e-8)

        return gradcam[0]  # (height, width) for the first sample

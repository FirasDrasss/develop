import math
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        patch_size: Tuple[int, int, int],
        hidden_dim: int,
    ):
        super().__init__()
        self.input_shape = input_shape
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim

        # Calculate number of patches
        self.num_patches = (
            (input_shape[0] // patch_size[0])
            * (input_shape[1] // patch_size[1])
            * (input_shape[2] // patch_size[2])
        )

        # Validate patch size
        if not all(i % p == 0 for i, p in zip(input_shape, patch_size)):
            raise ValueError(
                f"Input shape {input_shape} must be divisible by patch size {patch_size}"
            )

        # Calculate patch volume
        self.patch_volume = patch_size[0] * patch_size[1] * patch_size[2]

        # Simple linear projection for patch embedding
        self.proj = nn.Linear(self.patch_volume, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, 1, x, y, z]
        batch_size = x.shape[0]

        # Reshape to patches
        # [batch, 1, x, y, z] -> [batch, num_patches, patch_volume]
        x = x.unfold(2, self.patch_size[0], self.patch_size[0])  # unfold x
        x = x.unfold(3, self.patch_size[1], self.patch_size[1])  # unfold y
        x = x.unfold(4, self.patch_size[2], self.patch_size[2])  # unfold z

        # Combine patch dimensions
        x = x.reshape(batch_size, -1, self.patch_volume)

        # Project to hidden dimension and normalize
        x = self.proj(x)  # [batch, num_patches, hidden_dim]
        x = self.norm(x)

        return x


class NyquistPositionalEmbedding3D(nn.Module):
    """3D positional embedding using Nyquist frequency-based sine-cosine embeddings.

    Creates separate embeddings for x, y, z coordinates and combines them.
    Each dimension uses frequencies from 1/8 to Nyquist/(2*phi) to ensure smooth
    interpolation and avoid aliasing.

    The embedding for each position (x,y,z) is the concatenation of the embeddings
    for each coordinate, normalized to [0,1] range.
    """

    def __init__(
        self,
        d_model: int,
        input_shape: Tuple[int, int, int],
        patch_size: Tuple[int, int, int],
    ):
        """Initialize 3D positional embedding.

        Args:
            d_model: Total embedding dimension. Will be adjusted to nearest multiple of 3
                    and split equally among x,y,z.
            input_shape: Tuple of (x, y, z) dimensions of input.
            patch_size: Tuple of (x, y, z) dimensions for each patch.
        """
        super().__init__()

        # Adjust d_model to nearest multiple of 3
        self.d_model = d_model
        self.adjusted_dim = (
            (d_model + 2) // 3
        ) * 3  # Round up to nearest multiple of 3
        self.dim_per_axis = self.adjusted_dim // 3
        self.input_shape = input_shape
        self.patch_size = patch_size

        # Create embeddings for each axis
        self.embeddings = nn.ModuleList(
            [
                self._create_axis_embedding(dim, size)
                for dim, size in zip(["x", "y", "z"], input_shape)
            ]
        )

        # Projection to match original dimension if needed
        if self.adjusted_dim != d_model:
            self.proj: Union[nn.Linear, nn.Identity] = nn.Linear(
                self.adjusted_dim, d_model
            )
        else:
            self.proj: Union[nn.Linear, nn.Identity] = nn.Identity()

        # Final normalization layer
        self.norm = nn.LayerNorm(d_model)

    def _create_axis_embedding(self, axis: str, size: int) -> nn.Module:
        """Create embedding for a single axis using Nyquist frequencies."""
        k = self.dim_per_axis // 2  # Half for sine, half for cosine

        # Nyquist frequency for the axis (half the sampling rate)
        nyquist_frequency = size / 2

        # Use golden ratio to set max frequency
        golden_ratio = (1 + np.sqrt(5)) / 2
        frequencies = np.geomspace(
            1 / 8,  # Start with slow frequency
            nyquist_frequency / (2 * golden_ratio),  # End below Nyquist
            num=k,
        )

        # Create scale and bias for sine/cosine
        scale = np.repeat(2 * np.pi * frequencies, 2)
        bias = np.tile(np.array([0, np.pi / 2]), k)

        # Create module to hold the parameters
        embedding = nn.Module()
        embedding.register_buffer(
            "scale", torch.tensor(scale, dtype=torch.float32), persistent=False
        )
        embedding.register_buffer(
            "bias", torch.tensor(bias, dtype=torch.float32), persistent=False
        )
        return embedding

    def _get_normalized_positions(
        self, batch_size: int, num_patches: int
    ) -> torch.Tensor:
        """Get normalized positions for each patch in [0,1] range."""
        # Calculate patch grid dimensions
        patches_x = self.input_shape[0] // self.patch_size[0]
        patches_y = self.input_shape[1] // self.patch_size[1]
        patches_z = self.input_shape[2] // self.patch_size[2]

        # Create normalized positions for each axis
        x = torch.linspace(0, 1, patches_x)
        y = torch.linspace(0, 1, patches_y)
        z = torch.linspace(0, 1, patches_z)

        # Create meshgrid of positions
        pos_x, pos_y, pos_z = torch.meshgrid(x, y, z, indexing="ij")

        # Stack and reshape to [num_patches, 3]
        positions = torch.stack([pos_x, pos_y, pos_z], dim=-1)
        positions = positions.reshape(-1, 3)

        # Repeat for batch size
        positions = positions.unsqueeze(0).repeat(batch_size, 1, 1)

        return positions

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply positional embedding to input.

        Args:
            x: Input tensor of shape [batch, num_patches, d_model]

        Returns:
            Tensor of same shape with positional information added
        """
        batch_size = x.shape[0]
        num_patches = x.shape[1]

        # Get normalized positions for each patch
        positions = self._get_normalized_positions(batch_size, num_patches)

        # Get embeddings for each axis
        embeddings = []
        for i, embedding in enumerate(self.embeddings):
            # Get positions for this axis and add embedding dimension
            pos = positions[..., i].unsqueeze(-1)  # [batch, num_patches, 1]

            # Apply sine-cosine embedding
            emb = torch.addcmul(embedding.bias, embedding.scale, pos).sin()
            embeddings.append(emb)

        # Concatenate embeddings from all axes
        pos_emb = torch.cat(embeddings, dim=-1)  # [batch, num_patches, adjusted_dim]

        # Project to original dimension if needed
        pos_emb = self.proj(pos_emb)  # [batch, num_patches, d_model]

        # Normalize
        pos_emb = self.norm(pos_emb)

        # Add to input
        return x + pos_emb


class FlowFieldTransformer(nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        patch_size: Tuple[int, int, int] = (4, 4, 4),
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        expansion_factor: int = 4,
    ):
        """
        Transformer model for predicting 3D flow fields from boolean masks using patching.

        Args:
            input_shape: Tuple of (x, y, z) dimensions of input masks
            patch_size: Tuple of (x, y, z) dimensions for each patch
            hidden_dim: Dimension of the transformer hidden state
            num_layers: Number of transformer encoder layers
            num_heads: Number of attention heads
            dropout: Dropout probability
            expansion_factor: Factor to expand hidden dimension in feed-forward network

        Raises:
            ValueError: If input_shape is invalid or if num_heads doesn't divide hidden_dim
        """
        super().__init__()

        # Validate input shape
        if not isinstance(input_shape, tuple) or len(input_shape) != 3:
            raise ValueError(
                f"input_shape must be a tuple of 3 integers, got {input_shape}"
            )
        if not all(isinstance(dim, int) and dim > 0 for dim in input_shape):
            raise ValueError(
                f"All dimensions in input_shape must be positive integers, got {input_shape}"
            )

        # Validate hidden_dim and num_heads
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.input_shape = input_shape
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.num_channels = 5  # Flow field channels (u, v, w, p, t)

        # Patch embedding
        self.patch_embed = PatchEmbedding(input_shape, patch_size, hidden_dim)
        self.num_patches = self.patch_embed.num_patches

        # Positional encoding with Nyquist frequencies
        self.pos_encoder = NyquistPositionalEmbedding3D(
            hidden_dim, input_shape, patch_size
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * expansion_factor,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(
                hidden_dim,
                self.num_channels * patch_size[0] * patch_size[1] * patch_size[2],
            ),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _validate_input_shape(self, x: torch.Tensor) -> None:
        """
        Validate the input tensor shape.

        Args:
            x: Input tensor to validate

        Raises:
            ValueError: If input tensor shape is invalid
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Input must be a torch.Tensor, got {type(x)}")

        if x.dtype != torch.bool and x.dtype != torch.float32:
            raise ValueError(f"Input tensor must be bool or float32, got {x.dtype}")

        expected_shape = (None, *self.input_shape)  # None for batch dimension
        actual_shape = x.shape

        if len(actual_shape) != 4:
            raise ValueError(
                f"Input tensor must have 4 dimensions (batch, x, y, z), got shape {actual_shape}"
            )

        if actual_shape[1:] != self.input_shape:
            raise ValueError(
                f"Input tensor spatial dimensions {actual_shape[1:]} do not match "
                f"expected dimensions {self.input_shape}"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Input Processing Pipeline:
        1. Input Shape: (batch_size, x, y, z) - Boolean mask of obstacles
        2. Add channel dimension: (batch_size, 1, x, y, z)
        3. Patch Embedding:
           - Divides input into patches of size patch_size (e.g. 2x2x2)
           - For input (32,16,16) with patch_size (2,2,2):
             * Creates 16x8x8 = 1024 patches
             * Each patch contains 2x2x2 = 8 voxels
           - Projects each patch to hidden_dim using linear projection
           - Output: (batch_size, num_patches, hidden_dim)
        4. Positional Encoding:
           - Adds learned positional information to each patch
           - Maintains spatial relationships between patches
        5. Transformer Processing:
           - Processes patches through transformer layers
           - Each patch attends to all other patches
        6. Output Projection:
           - Projects each patch to num_channels * patch_size³
           - For patch_size (2,2,2) and 5 channels:
             * Each patch predicts 5 * 2 * 2 * 2 = 40 values
        7. Reshaping:
           - Rearranges predictions back to spatial dimensions
           - Final output: (batch_size, x, y, z, num_channels)
             where num_channels = (u, v, w, p, t) flow field values

        Args:
            x: Input tensor of shape (batch, x, y, z) containing boolean masks

        Returns:
            Output tensor of shape (batch, x, y, z, num_channels) containing flow field predictions

        Raises:
            ValueError: If input tensor shape is invalid
        """
        # Validate input shape
        self._validate_input_shape(x)

        batch_size = x.shape[0]

        # Convert bool to float if necessary
        if x.dtype == torch.bool:
            x = x.float()

        # Add channel dimension if needed
        if len(x.shape) == 4:
            x = x.unsqueeze(1)  # Add channel dimension: (batch, 1, x, y, z)

        # Patch embedding
        # Input: (batch, 1, x, y, z)
        # Output: (batch, num_patches, hidden_dim)
        # For (32,16,16) input with (2,2,2) patches:
        # num_patches = (32/2) * (16/2) * (16/2) = 16 * 8 * 8 = 1024
        # Each patch is flattened to 8 values (2*2*2) and projected to hidden_dim
        x = self.patch_embed(x)

        # Add positional encoding
        # Adds learned position information to each patch
        x = self.pos_encoder(x)

        # Apply transformer
        # Each patch attends to all other patches
        x = self.transformer_encoder(x)

        # Project to output channels
        # For patch_size (2,2,2) and 5 channels:
        # Each patch predicts 5 * 2 * 2 * 2 = 40 values
        x = self.output_projection(
            x
        )  # [batch, num_patches, num_channels * patch_size^3]

        # Reshape to spatial dimensions
        # First reshape to patch grid with channels and patch values
        x = x.reshape(
            batch_size,
            self.input_shape[0] // self.patch_size[0],  # 16
            self.input_shape[1] // self.patch_size[1],  # 8
            self.input_shape[2] // self.patch_size[2],  # 8
            self.num_channels,  # 5
            *self.patch_size,  # (2,2,2)
        )

        # Rearrange to final shape
        # 1. Permute to get channels and spatial dimensions in correct order
        # 2. Reshape to combine patch dimensions
        # 3. Final permute to get (batch, x, y, z, channels)
        x = x.permute(0, 4, 1, 5, 2, 6, 3, 7)  # [batch, num_channels, x, y, z]
        x = x.reshape(batch_size, self.num_channels, *self.input_shape)
        x = x.permute(0, 2, 3, 4, 1)  # [batch, x, y, z, num_channels]

        return x


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

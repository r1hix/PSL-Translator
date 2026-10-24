"""Bidirectional Long Short-Term Memory (Bi-LSTM) baseline model.

Academic Explainability:
    Serves as the mandatory academic reference baseline against which the proposed
    Spatial-Temporal Graph Convolutional Network (ST-GCN) is benchmarked.
    
    Architecture:
        1. Feature Flattening: Reshapes skeletal frame coordinates (V=55, C=3) -> 165-dim vector.
        2. Sequence Modeling: 2-layer Bidirectional LSTM capturing temporal gesture dynamics.
        3. Classification: Fully connected linear projection to 15 vocabulary logits.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn


class BaselineBiLSTM(nn.Module):
    """Bidirectional LSTM model for coordinate-space PSL sign classification."""

    def __init__(
        self,
        num_nodes: int = 55,
        in_channels: int = 3,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 15,
        dropout: float = 0.3,
    ) -> None:
        """Initializes Bi-LSTM layers and classification head.
        
        Args:
            num_nodes: Skeletal vertices V (default: 55).
            in_channels: Coordinate channels C (default: 3 for x, y, z).
            hidden_size: LSTM hidden state dimensionality.
            num_layers: Number of stacked LSTM layers.
            num_classes: Target sign vocabulary size (default: 15).
            dropout: Dropout probability between recurrent layers.
        """
        super().__init__()
        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.input_dim = num_nodes * in_channels  # 55 * 3 = 165

        # Normalization layer for numerical stability
        self.batch_norm = nn.BatchNorm1d(self.input_dim)

        # 2-layer Bidirectional LSTM
        # Bidirectional outputs shape (N, T, hidden_size * 2) = (N, 30, 256)
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)

        # Linear classifier: 256 -> 15 classes
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Bi-LSTM baseline.
        
        Args:
            x: Input batch tensor of shape (N, C, T, V).
            
        Returns:
            torch.Tensor: Logits tensor of shape (N, num_classes).
        """
        # Shape: (N, C, T, V) -> (N, T, V, C)
        n, c, t, v = x.size()
        x = x.permute(0, 2, 3, 1).contiguous()
        
        # Flatten spatial joints: (N, T, V * C) = (N, T, 165)
        x = x.view(n, t, v * c)

        # Apply batch norm along feature dimension: (N, 165, T)
        x = x.permute(0, 2, 1)
        x = self.batch_norm(x)
        x = x.permute(0, 2, 1)  # Return to (N, T, 165)

        # Pass through Bi-LSTM: output shape (N, T, 256)
        lstm_out, _ = self.lstm(x)

        # Temporal pooling: Take the mean state across all T=30 frames
        # Academic Explainability: Mean pooling across the temporal window
        # aggregates information from both gesture onset and climax.
        pooled = torch.mean(lstm_out, dim=1)  # Shape: (N, 256)

        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)      # Shape: (N, num_classes)
        return logits


if __name__ == "__main__":
    # Sanity check with dummy batch
    dummy_input = torch.randn(8, 3, 30, 55)
    model = BaselineBiLSTM(num_nodes=55, in_channels=3, num_classes=15)
    out = model(dummy_input)
    print("BaselineBiLSTM Forward Check:")
    print(f"  Input  : {dummy_input.shape} -> (N, C, T, V)")
    print(f"  Output : {out.shape} -> (N, num_classes)")
    print("  Model Parameter Count:", sum(p.numel() for p in model.parameters()))

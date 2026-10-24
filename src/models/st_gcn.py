"""Spatial-Temporal Graph Convolutional Network (ST-GCN) for Pakistani Sign Language.

Academic Explainability:
    Implements ST-GCN (Yan et al., AAAI 2018) tailored for 55-keypoint coordinate-space PSL:
    
    1. Spatial Graph Convolution:
       Models anatomical bone connectivity across 55 joints:
       f_out = sum_k [ (f_in * W_k) (A_k (x) M_k) ]
       where:
         - A_k in R^{V x V} is the normalized anatomical adjacency matrix for spatial partition k in {0, 1, 2}.
         - M_k in R^{V x V} is a learnable edge importance weight matrix initialized to 1.
         - W_k is the convolutional filter weight for spatial partition k.
         - (x) is the element-wise Hadamard product.
         
    2. Temporal Convolution (TCN):
       Captures continuous gesture velocity and rhythm across consecutive frames:
       Uses 2D convolutions with temporal receptive field kernel (9, 1).
       
    3. Residual Connections:
       Maintains gradient flow across deep spatial-temporal feature abstractions.
"""

from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.graph import PSLGraph


class SpatialGraphConv(nn.Module):
    """Spatial Graph Convolution Layer with learnable edge importance weighting."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        num_nodes: int = 55,
    ) -> None:
        """Initializes spatial graph convolution.
        
        Args:
            in_channels: Input joint feature channels C_in (e.g. 3 for x, y, z).
            out_channels: Output feature channels C_out.
            kernel_size: Spatial partition count K (typically 3 for root/centripetal/centrifugal).
            num_nodes: Skeletal vertices V (default: 55).
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.num_nodes = num_nodes
        self.out_channels = out_channels

        # Conv2d performs W_k transformation for all K partitions in parallel
        # Weight shape: (out_channels * kernel_size, in_channels, 1, 1)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(1, 1),
            padding=(0, 0),
            bias=False,
        )

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """Forward pass of spatial graph convolution.
        
        Mathematical Formulation:
            X_transformed = Conv2d(X) -> (N, K * C_out, T, V)
            For each partition k:
                Y_k = X_k * (A_k (*) M_k)
            Y = sum_k Y_k -> (N, C_out, T, V)
            
        Args:
            x: Input tensor of shape (N, C_in, T, V).
            A: Effective adjacency tensor of shape (K, V, V) incorporating learnable edge weights.
            
        Returns:
            torch.Tensor: Convolved feature tensor of shape (N, C_out, T, V).
        """
        n, c, t, v = x.size()
        
        # Linear projection: (N, C_in, T, V) -> (N, K * C_out, T, V)
        x_proj = self.conv(x)
        
        # Reshape to separate spatial partition dimension K: (N, K, C_out, T, V)
        x_proj = x_proj.view(n, self.kernel_size, self.out_channels, t, v)

        # Graph matrix multiplication:
        # Contraction over vertices dimension: Y_{n, c, t, w} = sum_{k, v} X_{n, k, c, t, v} * A_{k, v, w}
        # Academic Explainability: Aggregates joint features from neighboring anatomical bones.
        out = torch.einsum("nkctv,kvw->nctw", x_proj, A)
        return out.contiguous()


class STGCNBlock(nn.Module):
    """Unified Spatial-Temporal Graph Convolutional Block with Residual Connection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        num_nodes: int = 55,
        temporal_kernel_size: int = 9,
        stride: int = 1,
        dropout: float = 0.3,
    ) -> None:
        """Initializes Spatial Conv + Temporal Conv block with residual pathway.
        
        Args:
            in_channels: Input channels C_in.
            out_channels: Output channels C_out.
            kernel_size: Spatial partition count K (3).
            num_nodes: Skeletal vertices V (55).
            temporal_kernel_size: Temporal 1D receptive field span (default: 9 frames = 0.3s).
            stride: Temporal downsampling stride (1 or 2).
            dropout: Dropout probability.
        """
        super().__init__()
        
        # 1. Spatial Graph Convolution
        self.sgcn = SpatialGraphConv(in_channels, out_channels, kernel_size, num_nodes)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # 2. Temporal Convolutional Network (TCN)
        # Evaluates joint velocities and gesture trajectory rhythm across time
        padding = (temporal_kernel_size - 1) // 2
        self.tcn = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=(temporal_kernel_size, 1),
                stride=(stride, 1),
                padding=(padding, 0),
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout),
        )

        # 3. Residual Shortcut Connection
        # If channel count changes or temporal dimension is downsampled (stride > 1),
        # apply a 1x1 convolution to match shapes; otherwise use identity.
        if in_channels != out_channels or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=(1, 1),
                    stride=(stride, 1),
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """Forward pass through ST-GCN block.
        
        Args:
            x: Input tensor of shape (N, C_in, T, V).
            A: Adjacency matrix of shape (K, V, V).
            
        Returns:
            torch.Tensor: Convolved tensor of shape (N, C_out, T_out, V).
        """
        res = self.residual(x)
        
        # Spatial graph convolution + normalization + non-linearity
        x = self.relu(self.bn1(self.sgcn(x, A)))
        
        # Temporal convolution
        x = self.tcn(x)
        
        # Residual fusion
        x = self.relu(x + res)
        return x


class STGCN(nn.Module):
    """Complete Spatial-Temporal Graph Convolutional Network for PSL translation."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 15,
        num_nodes: int = 55,
        graph_strategy: str = "spatial",
        edge_importance_weighting: bool = True,
        dropout: float = 0.3,
    ) -> None:
        """Constructs ST-GCN backbone, edge weighting parameters, and classification head.
        
        Args:
            in_channels: Input channels C (3 for normalized x, y, z).
            num_classes: Target sign vocabulary classes (default: 15).
            num_nodes: Skeletal vertices V (default: 55).
            graph_strategy: Adjacency partitioning strategy ('spatial' for K=3).
            edge_importance_weighting: Learnable edge weighting mask M.
            dropout: Dropout rate across network.
        """
        super().__init__()
        
        # 1. Initialize Skeletal Graph Topology
        self.graph = PSLGraph(num_nodes=num_nodes, strategy=graph_strategy)
        # Register fixed anatomical adjacency as a non-trainable buffer
        self.register_buffer("A", torch.from_numpy(self.graph.A))  # Shape: (K, 55, 55)

        kernel_size = self.graph.A.shape[0]  # K = 3
        self.edge_importance_weighting = edge_importance_weighting

        # 2. Learnable Edge Importance Weights M
        # Academic Explainability: Allows the network to adaptively strengthen
        # significant finger/arm connections (e.g. index-to-thumb for pinching signs)
        # while dampening irrelevant bone links.
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.graph.A.shape))
                for _ in range(6)  # One parameter per ST-GCN block
            ])
        else:
            self.edge_importance = [1.0] * 6

        # 3. Input Data Normalization across (C * V) coordinates
        self.data_bn = nn.BatchNorm1d(in_channels * num_nodes)

        # 4. ST-GCN Backbone Blocks
        # Progressive channel expansion: 3 -> 64 -> 128 -> 256
        # Blocks 3 and 5 downsample the temporal window: T=30 -> T=15 -> T=8
        self.block1 = STGCNBlock(in_channels, 64, kernel_size, num_nodes, stride=1, dropout=dropout)
        self.block2 = STGCNBlock(64, 64, kernel_size, num_nodes, stride=1, dropout=dropout)
        self.block3 = STGCNBlock(64, 128, kernel_size, num_nodes, stride=2, dropout=dropout)
        self.block4 = STGCNBlock(128, 128, kernel_size, num_nodes, stride=1, dropout=dropout)
        self.block5 = STGCNBlock(128, 256, kernel_size, num_nodes, stride=2, dropout=dropout)
        self.block6 = STGCNBlock(256, 256, kernel_size, num_nodes, stride=1, dropout=dropout)

        # 5. Global Spatial-Temporal Pooling & Classification Head
        # Flattens spatial graph and temporal sequence into a unified 256-dim embedding
        self.fc = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of ST-GCN.
        
        Args:
            x: Input batch tensor of shape (N, C, T, V).
            
        Returns:
            torch.Tensor: Unnormalized class logits of shape (N, num_classes).
        """
        n, c, t, v = x.size()

        # Input Normalization: (N, C, T, V) -> (N, C * V, T)
        x_flat = x.permute(0, 1, 3, 2).contiguous().view(n, c * v, t)
        x_norm = self.data_bn(x_flat)
        x = x_norm.view(n, c, v, t).permute(0, 1, 3, 2).contiguous()

        # Sequential ST-GCN feature extraction
        # Each block applies its respective learnable edge importance weights: A_eff = A * M_i
        A = self.A
        x = self.block1(x, A * self.edge_importance[0])
        x = self.block2(x, A * self.edge_importance[1])
        x = self.block3(x, A * self.edge_importance[2])
        x = self.block4(x, A * self.edge_importance[3])
        x = self.block5(x, A * self.edge_importance[4])
        x = self.block6(x, A * self.edge_importance[5])

        # Global Average Pooling across both Time (T) and Joints (V):
        # Shape: (N, 256, T_out, V) -> (N, 256, 1, 1) -> (N, 256)
        x = F.avg_pool2d(x, kernel_size=(x.size(2), x.size(3)))
        x = x.view(n, -1)
        x = self.dropout(x)

        # Classification logits
        logits = self.fc(x)
        return logits


def build_model(
    architecture: str = "st_gcn",
    num_classes: int = 15,
    num_nodes: int = 55,
    in_channels: int = 3,
    dropout: float = 0.3,
) -> nn.Module:
    """Factory function initializing the requested neural architecture.
    
    Args:
        architecture: 'st_gcn' or 'baseline_lstm'.
        num_classes: Number of target sign classes.
        num_nodes: Skeletal vertices V (55).
        in_channels: Coordinate dimension C (3).
        dropout: Dropout rate.
        
    Returns:
        nn.Module: Initialized PyTorch model instance.
    """
    if architecture == "st_gcn":
        return STGCN(
            in_channels=in_channels,
            num_classes=num_classes,
            num_nodes=num_nodes,
            graph_strategy="spatial",
            edge_importance_weighting=True,
            dropout=dropout,
        )
    elif architecture == "baseline_lstm":
        from src.models.baseline_lstm import BaselineBiLSTM
        return BaselineBiLSTM(
            num_nodes=num_nodes,
            in_channels=in_channels,
            hidden_size=128,
            num_layers=2,
            num_classes=num_classes,
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unknown architecture: '{architecture}'")


if __name__ == "__main__":
    dummy = torch.randn(8, 3, 30, 55)
    model = build_model("st_gcn", num_classes=15)
    out = model(dummy)
    print("ST-GCN Forward Pass Verification:")
    print(f"  Input Batch Tensor : {dummy.shape} -> (N, C, T, V)")
    print(f"  Output Logits      : {out.shape} -> (N, num_classes)")
    print(f"  Total Parameters   : {sum(p.numel() for p in model.parameters()):,}")

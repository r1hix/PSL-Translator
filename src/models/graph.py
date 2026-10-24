"""Skeletal graph topology and spatial adjacency formulation for Pakistani Sign Language.

Academic Explainability:
    Formulates the dynamic human sign skeleton as an undirected spatial graph G = (V, E)
    with V = 55 vertices and E anatomical bone connections.
    
    Spatial Partitioning Strategy (Yan et al., AAAI 2018):
        Partitions the local 1-hop neighborhood into K=3 subsets based on anatomical
        proximity to the skeletal root joint (mid-shoulder chest center):
        - Partition 0 (Root / Self): Connects a node to itself (Identity matrix I).
        - Partition 1 (Centripetal): Connects nodes closer to the chest root.
        - Partition 2 (Centrifugal): Connects nodes further away toward extremities.
"""

from typing import List, Optional, Tuple
import numpy as np

# Anatomical Node Index Mapping (55 total nodes)
# 0-20  : Left Hand (21 nodes)
# 21-41 : Right Hand (21 nodes)
# 42-54 : Upper Body Pose (13 nodes)
ROOT_CHEST_NODE = 47  # Mid-shoulder region (between 47 L-Shoulder and 48 R-Shoulder)

# Intra-Hand Bone Topology (21 joints, relative to wrist index 0)
_HAND_PAIRS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index finger
    (0, 9), (9, 10), (10, 11), (11, 12),   # Middle finger
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring finger
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky finger
    (5, 9), (9, 13), (13, 17),             # Knuckle base arch
]

# Left Hand Bones (0 to 20)
_LEFT_HAND_BONES = [(u, v) for u, v in _HAND_PAIRS]

# Right Hand Bones (21 to 41)
_RIGHT_HAND_BONES = [(u + 21, v + 21) for u, v in _HAND_PAIRS]

# Upper Body Pose Bones (42 to 54)
# 42:Nose, 43:LEye, 44:REye, 45:LEar, 46:REar,
# 47:LShoulder, 48:RShoulder, 49:LElbow, 50:RElbow,
# 51:LWrist, 52:RWrist, 53:LHip, 54:RHip
_POSE_BONES = [
    (42, 43), (43, 45),                    # Left cranial
    (42, 44), (44, 46),                    # Right cranial
    (47, 48),                              # Clavicle / shoulder axis
    (47, 49), (49, 51),                    # Left humerus & radius/ulna
    (48, 50), (50, 52),                    # Right humerus & radius/ulna
    (47, 53), (48, 54), (53, 54),          # Pelvic / torso frame
]

# Bridge Connections: Links pose arm extremities to hand wrist roots
_BRIDGE_BONES = [
    (51, 0),   # Left Pose Wrist -> Left Hand Wrist Root
    (52, 21),  # Right Pose Wrist -> Right Hand Wrist Root
]

# Complete Anatomical Bone Connectivity (Undirected Edges)
SKELETAL_EDGES: List[Tuple[int, int]] = (
    _LEFT_HAND_BONES + _RIGHT_HAND_BONES + _POSE_BONES + _BRIDGE_BONES
)


def get_hop_distance(num_nodes: int, edges: List[Tuple[int, int]], max_hop: int = 1) -> np.ndarray:
    """Computes shortest graph path distances (hop counts) between all node pairs.
    
    Args:
        num_nodes: Total skeletal vertices V (55).
        edges: List of undirected bone index pairs.
        max_hop: Maximum neighborhood expansion radius (default: 1).
        
    Returns:
        np.ndarray: Symmetric distance matrix D of shape (V, V) where D_ij = hop count.
    """
    # Initialize with infinity
    dist = np.full((num_nodes, num_nodes), np.inf, dtype=np.float32)
    np.fill_diagonal(dist, 0)

    # 1-hop direct neighbors
    for u, v in edges:
        dist[u, v] = 1.0
        dist[v, u] = 1.0

    # Floyd-Warshall shortest path algorithm
    for k in range(num_nodes):
        for i in range(num_nodes):
            for j in range(num_nodes):
                if dist[i, k] + dist[k, j] < dist[i, j]:
                    dist[i, j] = dist[i, k] + dist[k, j]

    return dist


class PSLGraph:
    """Skeletal graph formulation for 55-keypoint Spatial-Temporal Graph Convolutions."""

    def __init__(
        self,
        num_nodes: int = 55,
        strategy: str = "spatial",
        max_hop: int = 1,
    ) -> None:
        """Initializes skeletal graph structure and computes partitioned adjacency matrices.
        
        Args:
            num_nodes: Vertex count V (55).
            strategy: Partitioning strategy ('spatial', 'uniform', or 'distance').
            max_hop: Graph convolutional receptive field radius (1-hop).
        """
        self.num_nodes = num_nodes
        self.strategy = strategy
        self.max_hop = max_hop
        self.edges = SKELETAL_EDGES

        # Compute hop distance matrix
        self.hop_dist = get_hop_distance(self.num_nodes, self.edges, max_hop=self.max_hop)

        # Build adjacency tensor A of shape (K, V, V)
        self.A = self._build_adjacency_matrix()

    def _build_adjacency_matrix(self) -> np.ndarray:
        """Builds normalized partitioned adjacency matrices based on the spatial strategy.
        
        Returns:
            np.ndarray: Adjacency tensor of shape (K, V, V) where K is partition count.
        """
        # Distance to gravity / root chest center (node 47 / 48)
        # We calculate shortest path hops from each node to root chest node 47
        root_dist = self.hop_dist[ROOT_CHEST_NODE]

        if self.strategy == "spatial":
            # K = 3 Spatial Configuration Partitioning (Yan et al., 2018)
            num_subsets = 3
            A = np.zeros((num_subsets, self.num_nodes, self.num_nodes), dtype=np.float32)

            for i in range(self.num_nodes):
                for j in range(self.num_nodes):
                    if self.hop_dist[i, j] <= self.max_hop:
                        if self.hop_dist[i, j] == 0:
                            # Partition 0: Self-connection (Identity)
                            A[0, i, j] = 1.0
                        elif root_dist[j] < root_dist[i]:
                            # Partition 1: Centripetal (closer to chest root)
                            A[1, i, j] = 1.0
                        else:
                            # Partition 2: Centrifugal (further toward extremities)
                            A[2, i, j] = 1.0

            # Degree-normalization: D^-1 * A for each partition
            for k in range(num_subsets):
                deg = np.sum(A[k], axis=1)
                deg_inv = np.zeros_like(deg)
                valid = deg > 0
                deg_inv[valid] = 1.0 / deg[valid]
                D_inv = np.diag(deg_inv)
                A[k] = np.dot(D_inv, A[k])

            return A.astype(np.float32)

        elif self.strategy == "uniform":
            # K = 1 Uniform Partitioning (Standard GCN normalized laplacian)
            A = np.zeros((1, self.num_nodes, self.num_nodes), dtype=np.float32)
            A[0] = np.where(self.hop_dist <= self.max_hop, 1.0, 0.0)

            # Degree normalization: D^-1/2 * A * D^-1/2
            deg = np.sum(A[0], axis=1)
            deg_inv_sqrt = np.zeros_like(deg)
            valid = deg > 0
            deg_inv_sqrt[valid] = 1.0 / np.sqrt(deg[valid])
            D_inv_sqrt = np.diag(deg_inv_sqrt)
            A[0] = np.dot(np.dot(D_inv_sqrt, A[0]), D_inv_sqrt)

            return A.astype(np.float32)

        else:
            raise ValueError(f"Unknown graph partitioning strategy: '{self.strategy}'")


if __name__ == "__main__":
    graph = PSLGraph(num_nodes=55, strategy="spatial")
    print(f"PSL Graph initialized successfully.")
    print(f"  Nodes (V)      : {graph.num_nodes}")
    print(f"  Edges (E)      : {len(graph.edges)}")
    print(f"  Adjacency Shape: {graph.A.shape} (K=3 spatial subsets)")

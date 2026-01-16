import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Graph:
    def __init__(self, strategy='spatial'):
        self.num_node = 17
        self.self_link = [(i, i) for i in range(self.num_node)]
        
        # COCO 17 Keypoints connections
        self.neighbor_link = [
            (0, 1), (0, 2), (1, 3), (2, 4), # Head
            (5, 6), # Shoulders
            (5, 7), (7, 9), # Left Arm
            (6, 8), (8, 10), # Right Arm
            (5, 11), (6, 12), # Torso
            (11, 12), # Hips
            (11, 13), (13, 15), # Left Leg
            (12, 14), (14, 16)  # Right Leg
        ]
        self.edge = self.self_link + self.neighbor_link
        self.center = 0 
        
        if strategy == 'spatial':
            self.A = self.get_spatial_graph()
        else:
            raise ValueError("Strategy must be 'spatial'")

    def get_spatial_graph(self):
        # Create adjacency matrix
        A = np.zeros((self.num_node, self.num_node))
        for i, j in self.edge:
            A[j, i] = 1
            A[i, j] = 1
        
        # Normalize
        DA = np.zeros((self.num_node, self.num_node))
        for i in range(self.num_node):
            if A[i].sum() > 0:
                DA[i, i] = (A[i].sum() + 1e-6) ** -1
        
        # This is a simplified spatial strategy (Uniform) for robustness
        # Ideally we partition neighbors (root, close, far) but uniform is often enough for general tasks
        norm_A = np.dot(DA, A)
        return torch.tensor(norm_A, dtype=torch.float32)

class ConvTemporalGraphical(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, t_kernel=1, stride=1, dilation=1, residue=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel, 1),
            padding=(t_kernel // 2, 0),
            stride=(stride, 1),
            dilation=(dilation, 1),
            bias=True)

    def forward(self, x, A):
        # x: (N, C, T, V)
        # A: (kernel_size, V, V) or (V, V)
        
        N, C, T, V = x.size()
        x = self.conv(x) # (N, out_C * K, T, V)
        
        # Reshape to mul with A
        x = x.view(N, self.kernel_size, -1, T, V)
        x = torch.einsum('nkctv,kvw->nctw', x, A)
        return x.contiguous()

class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dropout=0):
        super().__init__()
        
        # Spatial Graph Conv
        self.sgc = ConvTemporalGraphical(in_channels, out_channels, kernel_size=1) # A will have size 1 (Uniform) or 3 (Spatial)
        
        # Learnable Adjacency Importance
        self.M = nn.Parameter(torch.ones(1, 17, 17))

        # Temporal Conv
        self.tgc = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                (9, 1), # Temporal kernel size 9
                (stride, 1),
                ((9 - 1) // 2, 0),
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
            nn.ReLU()
        )
        
        # Residual
        if in_channels != out_channels or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.residual = nn.Identity()

        self.relu = nn.ReLU()

    def forward(self, x, A):
        res = self.residual(x)
        
        # Apply Importance to Adjacency
        A = A * self.M
        
        x = self.sgc(x, A)
        x = self.tgc(x)
        x = x + res
        x = self.relu(x)
        return x

class STGCN(nn.Module):
    def __init__(self, in_channels=3, num_class=2, edge_importance_weighting=True, **kwargs):
        super().__init__()

        # Graph
        self.graph = Graph()
        # Shape: (1, 17, 17) -> We treat it as 1 partition (uniform) for simplicity or expand if strategy changes
        # But `ConvTemporalGraphical` as written expects A to match kernel_size dim 0.
        # If we use 1 partition, kernel_size for ConvTemporalGraphical should be 1.
        self.A = self.graph.A.unsqueeze(0) # (1, V, V)
        self.register_buffer('A_buffer', self.A)

        # ST-GCN Network
        # Stack layers
        self.data_bn = nn.BatchNorm1d(in_channels * 17)
        
        self.st_gcn_networks = nn.ModuleList((
            STGCNBlock(in_channels, 64, kernel_size=1, stride=1),
            STGCNBlock(64, 64, kernel_size=1, stride=1),
            STGCNBlock(64, 64, kernel_size=1, stride=1),
            STGCNBlock(64, 128, kernel_size=1, stride=2),
            STGCNBlock(128, 128, kernel_size=1, stride=1),
            STGCNBlock(128, 128, kernel_size=1, stride=1),
            STGCNBlock(128, 256, kernel_size=1, stride=2),
            STGCNBlock(256, 256, kernel_size=1, stride=1),
            STGCNBlock(256, 256, kernel_size=1, stride=1),
        ))

        # Classification
        self.fcn = nn.Conv2d(256, num_class, kernel_size=1)

    def forward(self, x):
        # x: (N, C, T, V, M)
        N, C, T, V, M = x.size()
        
        # Permute to (N, M, C, T, V) -> (N*M, C, T, V)
        x = x.permute(0, 4, 1, 2, 3).contiguous().view(N * M, C, T, V)
        
        # Data BN (Input Norm)
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N * M, C, T, V)
        
        # Forward GCN
        for gcn in self.st_gcn_networks:
            x = gcn(x, self.A_buffer)
            
        # Global Pooling
        # x: (N*M, 256, T_out, V)
        x = F.avg_pool2d(x, x.size()[2:]) # Pool over spatio-temporal
        x = x.view(N, M, -1, 1, 1).mean(dim=1) # Mean over persons (M)
        
        # Prediction
        x = self.fcn(x) # (N, num_class, 1, 1)
        x = x.view(x.size(0), -1)
        
        return x

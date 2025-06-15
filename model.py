import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class GLU(nn.Module):
    """Gated Linear Unit with parameter sharing"""
    
    def __init__(self, input_dim: int, output_dim: int):
        super(GLU, self).__init__()
        self.gate_linear = nn.Linear(input_dim, output_dim * 2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_output = self.gate_linear(x)
        left, right = gate_output.chunk(2, dim=-1)
        return left * torch.sigmoid(right)


class SpectralConvLayer(nn.Module):
    """Spectral Convolution Layer with efficient FFT operations"""
    
    def __init__(self, time_steps: int, channels: int, kernel_size: int = 3):
        super(SpectralConvLayer, self).__init__()
        self.time_steps = time_steps
        self.channels = channels
        self.kernel_size = kernel_size
        
        # Learnable spectral filters
        self.spectral_conv = nn.Conv1d(
            in_channels=channels * 2,  # Real + Imaginary
            out_channels=channels * 2,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=2  # Separate processing for real/imaginary
        )
        
        self.activation = nn.GELU()
        self.norm = nn.LayerNorm(channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size, nodes, channels, time_steps]
        Returns:
            Processed tensor [batch_size, nodes, channels, time_steps]
        """
        batch_size, nodes, channels, time_steps = x.shape
        
        # Reshape for FFT: [batch_size * nodes, channels, time_steps]
        x_reshaped = x.view(batch_size * nodes, channels, time_steps)
        
        # Apply FFT
        x_fft = torch.fft.rfft(x_reshaped, dim=-1)
        
        # Split real and imaginary parts
        x_real = x_fft.real  # [batch_size * nodes, channels, freq_bins]
        x_imag = x_fft.imag  # [batch_size * nodes, channels, freq_bins]
        
        # Concatenate real and imaginary for convolution
        x_complex = torch.cat([x_real, x_imag], dim=1)  # [batch_size * nodes, channels*2, freq_bins]
        
        # Apply spectral convolution
        x_conv = self.spectral_conv(x_complex)
        x_conv = self.activation(x_conv)
        
        # Split back to real and imaginary
        channels_half = x_conv.size(1) // 2
        x_real_conv = x_conv[:, :channels_half]
        x_imag_conv = x_conv[:, channels_half:]
        
        # Reconstruct complex tensor
        x_complex_recon = torch.complex(x_real_conv, x_imag_conv)
        
        # Apply inverse FFT
        x_ifft = torch.fft.irfft(x_complex_recon, n=time_steps, dim=-1)
        
        # Reshape back: [batch_size, nodes, channels, time_steps]
        x_output = x_ifft.view(batch_size, nodes, channels, time_steps)
        
        # Apply normalization
        x_output = x_output.permute(0, 1, 3, 2)  # [batch_size, nodes, time_steps, channels]
        x_output = self.norm(x_output)
        x_output = x_output.permute(0, 1, 3, 2)  # [batch_size, nodes, channels, time_steps]
        
        return x_output


class GraphAttention(nn.Module):
    """Adaptive graph attention mechanism"""
    
    def __init__(self, feature_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super(GraphAttention, self).__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        
        assert feature_dim % num_heads == 0, "feature_dim must be divisible by num_heads"
        
        self.query_proj = nn.Linear(feature_dim, feature_dim)
        self.key_proj = nn.Linear(feature_dim, feature_dim)
        self.value_proj = nn.Linear(feature_dim, feature_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(feature_dim)
        
        # Learnable temperature parameter
        self.temperature = nn.Parameter(torch.ones(1) * (self.head_dim ** 0.5))
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input features [batch_size, nodes, time_steps, feature_dim]
        Returns:
            output: Attention output [batch_size, nodes, time_steps, feature_dim]
            adj_matrix: Learned adjacency matrix [batch_size, nodes, nodes]
        """
        batch_size, nodes, time_steps, feature_dim = x.shape
        
        # Aggregate temporal information for graph learning
        x_temporal = x.mean(dim=2)  # [batch_size, nodes, feature_dim]
        
        # Multi-head attention for graph structure learning
        q = self.query_proj(x_temporal).view(batch_size, nodes, self.num_heads, self.head_dim)
        k = self.key_proj(x_temporal).view(batch_size, nodes, self.num_heads, self.head_dim)
        
        # Compute attention scores
        attention_scores = torch.einsum('bnhd,bmhd->bhnm', q, k) / self.temperature
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Aggregate across heads to get adjacency matrix
        adj_matrix = attention_weights.mean(dim=1)  # [batch_size, nodes, nodes]
        
        # Apply attention to original features
        x_reshaped = x.view(batch_size, nodes, time_steps * feature_dim)
        attended_features = torch.einsum('bnm,bmt->bnt', adj_matrix, x_reshaped)
        attended_features = attended_features.view(batch_size, nodes, time_steps, feature_dim)
        
        # Residual connection and layer norm
        output = self.layer_norm(x + attended_features)
        
        return output, adj_matrix


class StemGNNBlock(nn.Module):
    """StemGNN block combining spectral and temporal modeling"""
    
    def __init__(self, 
                 time_steps: int, 
                 feature_dim: int, 
                 num_heads: int = 8,
                 dropout: float = 0.1):
        super(StemGNNBlock, self).__init__()
        
        self.spectral_conv = SpectralConvLayer(time_steps, feature_dim)
        self.graph_attention = GraphAttention(feature_dim, num_heads, dropout)
        
        self.temporal_conv = nn.Conv1d(
            in_channels=feature_dim,
            out_channels=feature_dim,
            kernel_size=3,
            padding=1,
            groups=feature_dim  # Depthwise convolution for efficiency
        )
        
        self.glu = GLU(feature_dim, feature_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(feature_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor [batch_size, nodes, time_steps, feature_dim]
        Returns:
            output: Processed tensor [batch_size, nodes, time_steps, feature_dim]
            adj_matrix: Learned adjacency matrix [batch_size, nodes, nodes]
        """
        batch_size, nodes, time_steps, feature_dim = x.shape
        
        # Spectral convolution
        x_spectral = x.permute(0, 1, 3, 2)  # [batch_size, nodes, feature_dim, time_steps]
        x_spectral = self.spectral_conv(x_spectral)
        x_spectral = x_spectral.permute(0, 1, 3, 2)  # [batch_size, nodes, time_steps, feature_dim]
        
        # Graph attention
        x_graph, adj_matrix = self.graph_attention(x_spectral)
        
        # Temporal convolution
        x_temporal = x_graph.view(batch_size * nodes, time_steps, feature_dim)
        x_temporal = x_temporal.permute(0, 2, 1)  # [batch_size * nodes, feature_dim, time_steps]
        x_temporal = self.temporal_conv(x_temporal)
        x_temporal = x_temporal.permute(0, 2, 1)  # [batch_size * nodes, time_steps, feature_dim]
        x_temporal = x_temporal.view(batch_size, nodes, time_steps, feature_dim)
        
        # GLU activation and residual connection
        x_glu = self.glu(x_temporal)
        x_output = self.layer_norm(x + self.dropout(x_glu))
        
        return x_output, adj_matrix


class StemGNN(nn.Module):
    """StemGNN model for multivariate time series forecasting"""
    
    def __init__(self,
                 num_nodes: int,
                 input_dim: int = 1,
                 hidden_dim: int = 64,
                 output_dim: int = 1,
                 num_layers: int = 3,
                 num_heads: int = 8,
                 window_size: int = 12,
                 horizon: int = 3,
                 dropout: float = 0.1):
        super(StemGNN, self).__init__()
        
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.window_size = window_size
        self.horizon = horizon
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # StemGNN blocks
        self.stem_blocks = nn.ModuleList([
            StemGNNBlock(window_size, hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, horizon * output_dim)
        )
        
        # Initialize parameters
        self._init_parameters()
        
    def _init_parameters(self):
        """Initialize model parameters"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor [batch_size, window_size, num_nodes]
        Returns:
            forecast: Forecast tensor [batch_size, horizon, num_nodes]
            adj_matrix: Final adjacency matrix [batch_size, num_nodes, num_nodes]
        """
        batch_size, window_size, num_nodes = x.shape
        
        # Reshape and project input: [batch_size, num_nodes, window_size, hidden_dim]
        x = x.permute(0, 2, 1).unsqueeze(-1)  # [batch_size, num_nodes, window_size, 1]
        x = self.input_proj(x)  # [batch_size, num_nodes, window_size, hidden_dim]
        
        # Apply StemGNN blocks
        adj_matrices = []
        for block in self.stem_blocks:
            x, adj_matrix = block(x)
            adj_matrices.append(adj_matrix)
        
        # Use the last adjacency matrix
        final_adj_matrix = adj_matrices[-1]
        
        # Global average pooling over time dimension
        x = x.mean(dim=2)  # [batch_size, num_nodes, hidden_dim]
        
        # Output projection
        forecast = self.output_proj(x)  # [batch_size, num_nodes, horizon * output_dim]
        forecast = forecast.view(batch_size, num_nodes, self.horizon, self.output_dim)
        forecast = forecast.squeeze(-1).permute(0, 2, 1)  # [batch_size, horizon, num_nodes]
        
        return forecast, final_adj_matrix

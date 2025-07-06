import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

import numpy as np
import scipy
import matplotlib.pyplot as plt


from tonic.torch.agents.diffusion_utils.sigma_embeddings import *


class MLPNetwork(nn.Module):
    """
    Simple multi layer perceptron network which can be generated with different 
    activation functions with and without spectral normalization of the weights
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 100,
        num_hidden_layers: int = 1,
        output_dim=1,
        dropout: int = 0,
        activation: str = "ReLU",
        use_spectral_norm: bool = False,
        device: str = 'cuda'
    ):
        super(MLPNetwork, self).__init__()
        self.network_type = "mlp"
        # define number of variables in an input sequence
        self.input_dim = input_dim
        # the dimension of neurons in the hidden layer
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        # number of samples per batch
        self.output_dim = output_dim
        self.dropout = dropout
        self.spectral_norm = use_spectral_norm
        # set up the network
        self.layers = nn.ModuleList([nn.Linear(self.input_dim, self.hidden_dim)])
        self.layers.extend(
            [
                nn.Linear(self.hidden_dim, self.hidden_dim)
                for i in range(1, self.num_hidden_layers)
            ]
        )
        self.layers.append(nn.Linear(self.hidden_dim, self.output_dim))

        # build the activation layer
        self.act = return_activation_fcn(activation)
        self._device = device
        self.layers.to(self._device)

    def forward(self, x):

        for idx, layer in enumerate(self.layers):
            if idx == 0:
                out = layer(x)
            else:
                if idx < len(self.layers) - 2:
                    out = layer(out) # + out
                else:
                    out = layer(out)
            if idx < len(self.layers) - 1:
                out = self.act(out)
        return out

    def get_device(self, device: torch.device):
        self._device = device
        self.layers.to(device)
    
    def get_params(self):
        return self.layers.parameters()


def return_activation_fcn(activation_type: str):
    # build the activation layer
    if activation_type.lower() == "sigmoid":
        act = nn.Sigmoid()
    elif activation_type.lower() == "tanh":
        act = nn.Tanh()  # Fixed from Sigmoid to Tanh
    elif activation_type.lower() == "relu":
        act = nn.ReLU()
    elif activation_type.lower() == "prelu":
        act = nn.PReLU()
    elif activation_type.lower() == "softmax":
        act = nn.Softmax(dim=-1)
    elif activation_type.lower() == "mish":
        act = nn.Mish()
    elif activation_type.lower() == "gelu":
        act = nn.GELU()
    else:
        act = nn.PReLU()  # Default
    return act


class TwoLayerPreActivationResNetLinear(nn.Module):
    """
    A proper implementation of two-layer pre-activation ResNet block.
    Pre-activation means: normalize → activate → dropout → linear
    """
    def __init__(
            self,
            hidden_dim: int = 100,
            activation: str = 'relu',
            dropout_rate: float = 0.25,
            spectral_norm: bool = False,
            use_norm: bool = False,
            norm_style: str = 'BatchNorm'  # Fixed type to str
    ) -> None:
        super().__init__()
        # Create linear layers with or without spectral normalization
        if spectral_norm:
            self.l1 = nn.utils.spectral_norm(nn.Linear(hidden_dim, hidden_dim))
            self.l2 = nn.utils.spectral_norm(nn.Linear(hidden_dim, hidden_dim))
        else:
            self.l1 = nn.Linear(hidden_dim, hidden_dim)
            self.l2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Create dropout layer if needed
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
        self.use_norm = use_norm
        self.act = return_activation_fcn(activation)

        # Create normalization layers if needed
        if use_norm:
            if norm_style == 'BatchNorm':
                # Using BatchNorm1d properly
                self.norm1 = nn.BatchNorm1d(hidden_dim)
                self.norm2 = nn.BatchNorm1d(hidden_dim)
            elif norm_style == 'LayerNorm':
                self.norm1 = nn.LayerNorm(hidden_dim, eps=1e-06)
                self.norm2 = nn.LayerNorm(hidden_dim, eps=1e-06)
            else:
                raise ValueError(f'Unsupported normalization type: {norm_style}')

    def forward(self, x):
        # Save input for residual connection
        x_input = x
        
        # First pre-activation block
        if self.use_norm:
            if isinstance(self.norm1, nn.BatchNorm1d) and x.dim() == 2:
                # Handle BatchNorm1d for 2D inputs
                x = self.norm1(x)
            elif isinstance(self.norm1, nn.LayerNorm):
                x = self.norm1(x)
        
        # Activation
        x = self.act(x)
        
        # Dropout (if used)
        if self.dropout is not None:
            x = self.dropout(x)
            
        # First linear layer
        x = self.l1(x)
        
        # Second pre-activation block
        if self.use_norm:
            if isinstance(self.norm2, nn.BatchNorm1d) and x.dim() == 2:
                # Handle BatchNorm1d for 2D inputs
                x = self.norm2(x)
            elif isinstance(self.norm2, nn.LayerNorm):
                x = self.norm2(x)
                
        # Activation
        x = self.act(x)
        
        # Dropout (if used)
        if self.dropout is not None:
            x = self.dropout(x)
            
        # Second linear layer
        x = self.l2(x)
        
        # Residual connection
        return x + x_input


class ResidualMLPNetwork(nn.Module):
    """
    Simple multi-layer perceptron network with residual connections for 
    benchmarking the performance of different networks. The residual layers
    are based on the IBC paper implementation, which uses 2 residual layers
    with pre-activation with or without dropout and normalization.
    """
    def __init__( #in_dim=4, out_dim=2, hidden_dim=256, n_hidden=4, sigma_data=1.0
        self,
        in_dim: int= 4,
        out_dim: int = 2,
        hidden_dim: int = 256,
        n_hidden: int = 6,  # Changed default to 2 to satisfy the assertion
        sigma_data: float = 1.0,
        dropout: float = 0.0,  # Fixed type to float
        activation: str = "Mish",
        use_spectral_norm: bool = False,
        use_norm: bool = False,
        norm_style: str = 'BatchNorm',
        device: str = 'cpu',

    ):
        super(ResidualMLPNetwork, self).__init__()
        self.network_type = "mlp"
        self._device = device
        self.sigma_data = sigma_data
        
        # Ensure the number of hidden layers is even for residual blocks
        assert n_hidden % 2 == 0, "n_hidden must be even"
        
        # First layer: input_dim to hidden_dim
        if use_spectral_norm:
            self.layers = nn.ModuleList([nn.utils.spectral_norm(nn.Linear(in_dim, hidden_dim))])
        else:
            self.layers = nn.ModuleList([nn.Linear(in_dim, hidden_dim)])
        
        # Add activation after the first layer
        self.first_activation = return_activation_fcn(activation)
        
        # Add residual blocks (each handles 2 layers)
        self.layers.extend(
            [
                TwoLayerPreActivationResNetLinear(
                    hidden_dim=hidden_dim,
                    activation=activation,
                    dropout_rate=dropout,
                    spectral_norm=use_spectral_norm,
                    use_norm=use_norm,
                    norm_style=norm_style
                )
                for i in range(n_hidden // 2)  # Each block counts as 2 layers
            ]
        )
        
        # Output layer
        self.layers.append(nn.Linear(hidden_dim, out_dim))
        
        # Move to device
        self.to(self._device)

    def forward(self, inp, state):
        # First layer
        x = torch.cat([inp, state.to(inp.device)], dim=-1)
        x = self.layers[0](x.to(torch.float32))
        x = self.first_activation(x)
        
        # Residual blocks and output layer
        for idx, layer in enumerate(self.layers[1:], 1):
            x = layer(x)
            
        return x

    def get_device(self, device: torch.device):
        self._device = device
        self.to(device)
    
    def get_params(self):
        return self.parameters()  # More standard way to get parameters


    
# Modified to include simple scalar condition (0-1)
class ConditionalMLP(nn.Module):
    def __init__(self,in_dim=4, out_dim=2, hidden_dim=256, embed_dim=32, embed_type='Sinusoidal', n_hidden=4, sigma_data=1.0):
        super().__init__()

        self.sigma_data = sigma_data

    
        # Main network with additional input for condition
        layers = []
        # 4 = 2 (data dimension) + 1 (noise level) + 1 (condition)
      
        
        # First layer processes data, noise level, and condition directly
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.Mish())
        
        # Hidden layers
        for _ in range(n_hidden - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Mish())
        
        # Output layer
        last_linear = nn.Linear(hidden_dim, out_dim)
        nn.init.zeros_(last_linear.weight)
        nn.init.zeros_(last_linear.bias)
        layers.append(last_linear)
        


        self.network = nn.Sequential(*layers)
        
        
        if embed_type == 'sinusoidal':
            self.sigma_embed = nn.Sequential(
                SinusoidalProjection(embed_dim),
                nn.Linear(embed_dim, embed_dim * 4),
                nn.Mish(),
                nn.Linear(embed_dim * 4, embed_dim),
                )
        elif embed_type == 'fourier':
            self.sigma_embed = nn.Sequential(
                FourierFeaturesEmbedding(embed_dim),
                nn.Linear(embed_dim, embed_dim * 4),
                nn.Mish(),
                nn.Linear(embed_dim * 4, embed_dim),
                )
        else:
            ValueError("\n Sigma Embeddings are not assigned correctly within ConditionalMLP \n")
            

    def forward(self, x, sigma, states):
        # Concatenate condition directly with input
        # condition should be a scalar between 0 and 1
        sigma_emb = self.sigma_embed(sigma) 
        sigma_emb = sigma_emb.squeeze(1)
        
        inp = torch.cat([x, sigma_emb.to(x.device), states.to(x.device)], dim=-1)
    
        output = self.network(inp)
        return output
    

    

class EtaMLP(nn.Module):
    def __init__(self,in_dim, out_dim, hidden_dim=256, n_hidden=2):
        super().__init__()


        self.eta_eps = 1e-8
        # Main network with additional input for condition
        layers = []

        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.Mish())
        
        # Hidden layers
        for _ in range(n_hidden - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Mish())
        
        # Output layer
        last_linear = nn.Linear(hidden_dim, out_dim)
        nn.init.zeros_(last_linear.weight)
        nn.init.zeros_(last_linear.bias)
        layers.append(last_linear)
        
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):

        η_free = self.network(x)
        
        η = F.softplus(η_free) + self.eta_eps    # positive: η ∈ (ε, ∞)

        return η
        

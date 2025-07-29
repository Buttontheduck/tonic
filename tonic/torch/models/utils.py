import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Independent, MixtureSameFamily,MultivariateNormal, Normal
from typing import Sequence, Optional, Callable
import math



def uniform_initializer(layer , scale = 0.333333):
    fan_out = layer.weight.size(0)  # number of output features
    L = torch.sqrt(torch.tensor(3.0 * scale / fan_out))
    torch.nn.init.uniform_(layer.weight, -L, L)
    if layer.bias is not None:
        torch.nn.init.zeros_(layer.bias)



class LayerNormMLP(torch.nn.Module):
    def __init__(self,  
        layer_sizes: Sequence[int],
        activation: Callable[[torch.Tensor], torch.Tensor] = F.elu,
        activate_final: bool = False,):
        super().__init__()
        
        self.layer_sizes = layer_sizes
        self.w_initializer= uniform_initializer
        self.activation = activation
        self.activate_final = activate_final
        
    def initialize(self, input_size, device= "cpu"):
        
        if len(self.layer_sizes) < 1:
            raise ValueError("layer_sizes must have at least one element")

        layers = [] 
        
        layer0 = nn.Linear(input_size,self.layer_sizes[0])
        self.w_initializer(layer0)
        
        
        layers.append(layer0)
        layers.append(torch.nn.LayerNorm(normalized_shape=self.layer_sizes[0], elementwise_affine=True))
        layers.append(nn.Tanh())
        
        in_dim = self.layer_sizes[0]
        for out_dim in self.layer_sizes[1:]:
            layer_hidden =  nn.Linear(in_dim,out_dim)
            self.w_initializer(layer_hidden)
            layers.append(layer_hidden)
            layers.append(self.activation)
            in_dim = out_dim
            
        self.model = nn.Sequential(*layers)
        self.model = self.model.to(device)
        
        return self.layer_sizes[-1]
    
    def forward(self, inputs):
        return self.model(inputs)
         
         
class MLP(torch.nn.Module):
    def __init__(self, sizes, activation, fn=None):
        super().__init__()
        self.sizes = sizes
        self.activation = activation
        self.fn = fn        

    def initialize(self, input_size, device):
        sizes = [input_size] + list(self.sizes)
        layers = []
        for i in range(len(sizes) - 1):
            layers += [torch.nn.Linear(sizes[i], sizes[i + 1]),
                       self.activation()]
        self.model = torch.nn.Sequential(*layers)
        self.model = self.model.to(device)
        if self.fn is not None:
            self.model.apply(self.fn)
        return sizes[-1]

    def forward(self, inputs):
        return self.model(inputs)



def trainable_variables(model):
    return [p for p in model.parameters() if p.requires_grad]

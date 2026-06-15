import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm

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





class StableMLP(torch.nn.Module):
    def __init__(self, sizes, activation, fn=None, spectral_normalization=True):
        super().__init__()
        self.sizes = sizes
        self.activation = activation
        self.fn = fn
        self.spectral_normalization = spectral_normalization

    def initialize(self, input_size, device):
        sizes = [input_size] + list(self.sizes)
        layers = []
        for i in range(len(sizes) - 1):
            linear_layer = torch.nn.Linear(sizes[i], sizes[i + 1])
            
            if self.spectral_normalization:
                # Use the legacy function wrapper
                linear_layer = spectral_norm(linear_layer)
                
            layers += [linear_layer, self.activation()]
            
        self.model = torch.nn.Sequential(*layers)
        self.model = self.model.to(device)
        
        if self.fn is not None:
            self.model.apply(self.fn)
            
        return sizes[-1]

    def forward(self, inputs):
        return self.model(inputs)
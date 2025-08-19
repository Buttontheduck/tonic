import torch
from tonic.torch.agents.diffusion_utils.denoiser_networks import EtaMLP

class Temperature(torch.nn.Module):
    def __init__(self, encoder,hidden_dim, n_hidden ,device):
        super().__init__()
        self.encoder = encoder
        self.n_hidden = n_hidden       
        self.hidden_dim = hidden_dim
        self.device = device

    def initialize(
        self, observation_space):
        size = self.encoder.initialize(
            observation_space)
        self.model = EtaMLP(in_dim=size,out_dim=1,hidden_dim=self.hidden_dim,n_hidden=self.n_hidden).to(self.device)
        
    def forward(self, *inputs):
        out = self.model(*inputs)
        return out

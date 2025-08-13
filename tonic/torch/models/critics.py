import torch
import torch.nn.functional as F
from torch.distributions import Normal, MultivariateNormal, Categorical, MixtureSameFamily, Independent
from typing import Optional
from tonic.torch.models.utils import variance_scaling_init
_MIN_SCALE = 1e-6  

class ValueHead(torch.nn.Module):
    def __init__(self, fn=None):
        super().__init__()
        self.fn = fn

    def initialize(self, input_size, return_normalizer=None, device = 'cpu'):
        self.return_normalizer = return_normalizer
        self.v_layer = torch.nn.Linear(input_size, 1)
        self.v_layer = self.v_layer.to(device)
        if self.fn:
            self.v_layer.apply(self.fn)

    def forward(self, inputs):
        out = self.v_layer(inputs)
        out = torch.squeeze(out, -1)
        if self.return_normalizer:
            out = self.return_normalizer(out)
        return out


class CategoricalWithSupport:
    def __init__(self, values, logits):
        self.values = values
        self.logits = logits
        self.probabilities = torch.nn.functional.softmax(logits, dim=-1)

    def mean(self):
        return (self.probabilities.to("cpu") * self.values).sum(dim=-1)

    def project(self, returns):
        vmin, vmax = self.values[0], self.values[-1]
        d_pos = torch.cat([self.values, vmin[None]], 0)[1:]
        d_pos = (d_pos - self.values)[None, :, None]
        d_neg = torch.cat([vmax[None], self.values], 0)[:-1]
        d_neg = (self.values - d_neg)[None, :, None]

        clipped_returns = torch.clamp(returns, vmin, vmax)
        delta_values = clipped_returns[:, None] - self.values[None, :, None]
        delta_sign = (delta_values >= 0).float()
        delta_hat = ((delta_sign * delta_values / d_pos) -
                     ((1 - delta_sign) * delta_values / d_neg))
        delta_clipped = torch.clamp(1 - delta_hat, 0, 1)

        return (delta_clipped * self.probabilities[:, None].to("cpu")).sum(dim=2)


class DistributionalValueHead(torch.nn.Module):
    def __init__(self, vmin, vmax, num_atoms, fn=None):
        super().__init__()
        self.num_atoms = num_atoms
        self.fn = fn
        self.values = torch.linspace(vmin, vmax, num_atoms).float()

    def initialize(self, input_size, return_normalizer=None, device = 'cpu'):
        if return_normalizer:
            raise ValueError(
                'Return normalizers cannot be used with distributional value'
                'heads.')
        self.distributional_layer = torch.nn.Linear(input_size, self.num_atoms).to(device)
        if self.fn:
            self.distributional_layer.apply(self.fn)

    def forward(self, inputs):
        logits = self.distributional_layer(inputs)
        return CategoricalWithSupport(values=self.values, logits=logits)
    
    
    
    
    
class GaussianMixtureHead(torch.nn.Module):
    def __init__(self, num_dimensions: int, num_components: int, init_scale: Optional[float] = None):
        super().__init__()
        
        self.num_dimensions = num_dimensions
        self.num_components = num_components

        self.init_scale = init_scale

        if self.init_scale is not None:
            self.scale_factor = self.init_scale / F.softplus(torch.tensor(0.))
        else:
            self.scale_factor = 1.0 
               
        self.out_dim = self.num_components * self.num_dimensions

        self.logit_size = self.num_components * self.num_dimensions
            
        self.w_init  = lambda tensor: variance_scaling_init(tensor, scale=1e-5)
            
    def initialize(self, input_size,  device = 'cpu'):
        

        self.input_size = input_size   
        self.device = device
    

        self.logit_layer = torch.nn.Linear(self.input_size,self.logit_size).to(self.device)
        self.loc_layer   = torch.nn.Linear(self.input_size,self.out_dim).to(self.device)  # out_dim = self.num_components * self.num_dimensions
        self.scale_layer = torch.nn.Linear(self.input_size,self.out_dim).to(self.device)  # out_dim = self.num_components * self.num_dimensions
        
        variance_scaling_init(self.logit_layer.weight, scale=1e-5)
        variance_scaling_init(self.loc_layer.weight, scale=1e-5)
        variance_scaling_init(self.scale_layer.weight, scale=1e-5)
        
    def forward(self,inputs):
        
        batch_size = inputs.shape[0]
        
        logits = self.logit_layer(inputs)
        locs = self.loc_layer(inputs)
        

        scales = self.scale_layer(inputs)
        scales = self.scale_factor * F.softplus(scales) + _MIN_SCALE
        

     
        locs = locs.reshape(batch_size, self.num_dimensions, self.num_components)
        scales = scales.reshape(batch_size, self.num_dimensions, self.num_components)
        logits = logits.reshape(batch_size, self.num_dimensions, self.num_components)
            

        components = Normal(locs, scales)
        mixture = Categorical(logits=logits)
        
        return MixtureSameFamily(mixture, components)
            
            
        

class Critic(torch.nn.Module):
    def __init__(self, encoder, torso, head,device):
        super().__init__()
        self.encoder = encoder
        self.torso = torso
        self.head = head
        self.device = device

    def initialize(
        self, observation_space, action_space, observation_normalizer=None,
        return_normalizer=None
    ):
        size = self.encoder.initialize(
            observation_space=observation_space, action_space=action_space,
            observation_normalizer=observation_normalizer,device= self.device)
        size = self.torso.initialize(size,device=self.device)
        self.head.initialize(size, return_normalizer,device= self.device)

    def forward(self, *inputs):
        out = self.encoder(*inputs)
        out = self.torso(out)
        return self.head(out)

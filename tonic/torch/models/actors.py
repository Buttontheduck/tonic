import torch
import torch.nn as nn
import torch.nn.functional as F
from tonic.torch.agents.diffusion_utils.diffusion_agents.k_diffusion.gc_sampling import *
from tonic.torch.agents.diffusion_utils.diffusion_agents.k_diffusion.score_wrappers import GCDenoiser  as trainer
from functools import partial
import math
from tonic.torch.agents.diffusion_utils.denoiser_networks import ConditionalMLP, ResidualMLPNetwork
import scipy

FLOAT_EPSILON = 1e-8


class SquashedMultivariateNormalDiag:
    def __init__(self, loc, scale):
        self._distribution = torch.distributions.normal.Normal(loc, scale)

    def rsample_with_log_prob(self, shape=()):
        samples = self._distribution.rsample(shape)
        squashed_samples = torch.tanh(samples)
        log_probs = self._distribution.log_prob(samples)
        log_probs -= torch.log(1 - squashed_samples ** 2 + 1e-6)
        return squashed_samples, log_probs

    def rsample(self, shape=()):
        samples = self._distribution.rsample(shape)
        return torch.tanh(samples)

    def sample(self, shape=()):
        samples = self._distribution.sample(shape)
        return torch.tanh(samples)

    def log_prob(self, samples):
        '''Required unsquashed samples cannot be accurately recovered.'''
        raise NotImplementedError(
            'Not implemented to avoid approximation errors. '
            'Use sample_with_log_prob directly.')

    @property
    def loc(self):
        return torch.tanh(self._distribution.mean)


class DetachedScaleGaussianPolicyHead(torch.nn.Module):
    def __init__(
        self, loc_activation=torch.nn.Tanh, loc_fn=None, log_scale_init=0.,
        scale_min=1e-4, scale_max=1.,
        distribution=torch.distributions.normal.Normal
    ):
        super().__init__()
        self.loc_activation = loc_activation
        self.loc_fn = loc_fn
        self.log_scale_init = log_scale_init
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.distribution = distribution

    def initialize(self, input_size, action_size):
        self.loc_layer = torch.nn.Sequential(
            torch.nn.Linear(input_size, action_size), self.loc_activation())
        if self.loc_fn:
            self.loc_layer.apply(self.loc_fn)
        log_scale = [[self.log_scale_init] * action_size]
        self.log_scale = torch.nn.Parameter(
            torch.as_tensor(log_scale, dtype=torch.float32))

    def forward(self, inputs):
        loc = self.loc_layer(inputs)
        batch_size = inputs.shape[0]
        scale = torch.nn.functional.softplus(self.log_scale) + FLOAT_EPSILON
        scale = torch.clamp(scale, self.scale_min, self.scale_max)
        scale = scale.repeat(batch_size, 1)
        return self.distribution(loc, scale)


class GaussianPolicyHead(torch.nn.Module):
    def __init__(
        self, loc_activation=torch.nn.Tanh, loc_fn=None,
        scale_activation=torch.nn.Softplus, scale_min=1e-4, scale_max=1,
        scale_fn=None, distribution=torch.distributions.normal.Normal
    ):
        super().__init__()
        self.loc_activation = loc_activation
        self.loc_fn = loc_fn
        self.scale_activation = scale_activation
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.scale_fn = scale_fn
        self.distribution = distribution

    def initialize(self, input_size, action_size):
        self.loc_layer = torch.nn.Sequential(
            torch.nn.Linear(input_size, action_size), self.loc_activation())
        if self.loc_fn:
            self.loc_layer.apply(self.loc_fn)
        self.scale_layer = torch.nn.Sequential(
            torch.nn.Linear(input_size, action_size), self.scale_activation())
        if self.scale_fn:
            self.scale_layer.apply(self.scale_fn)

    def forward(self, inputs):
        loc = self.loc_layer(inputs)
        scale = self.scale_layer(inputs)
        scale = torch.clamp(scale, self.scale_min, self.scale_max)
        return self.distribution(loc, scale)


class DeterministicPolicyHead(torch.nn.Module):
    def __init__(self, activation=torch.nn.Tanh, fn=None):
        super().__init__()
        self.activation = activation
        self.fn = fn

    def initialize(self, input_size, action_size):
        self.action_layer = torch.nn.Sequential(
            torch.nn.Linear(input_size, action_size),
            self.activation())
        if self.fn is not None:
            self.action_layer.apply(self.fn)

    def forward(self, inputs):
        return self.action_layer(inputs)


class DiffusionPolicyHead(torch.nn.Module):
    def __init__(self, device="cpu", num_diffusion_steps=50, hidden_dim=256,embed_dim=32,embed_type ='fourier',n_hidden=4,n_blocks=6, sigma_data=1.0,sampler_type='ddim',model_type = 'mlp', \
        sigma_max=80, sigma_min=0.001, rho=7, noise_type = 'karras' ):
        
        self.device = device
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.embed_type = embed_type
        self.n_hidden=n_hidden
        self.sigma_data=sigma_data
        self.sampler_type=sampler_type
        self.n_blocks = n_blocks
        self.n_diffusion_steps = num_diffusion_steps
        self.model_type = model_type
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.rho = 7
        self.noise_type = noise_type
        
        


    def initialize(self, state_size, action_size):
        super(DiffusionPolicyHead, self).__init__()
        self.state_dim = state_size
        self.action_dim = action_size
        input_dim = self.state_dim + self.action_dim + self.embed_dim
        
        
        if self.model_type=='mlp':
            self.model = ConditionalMLP(in_dim=input_dim, out_dim=self.action_dim,hidden_dim=self.hidden_dim,embed_dim=self.embed_dim,embed_type=self.embed_type,n_hidden=self.n_hidden,sigma_data=self.sigma_data).to(self.device)
        elif self.model_type=='resmlp':
            self.model= ResidualMLPNetwork(in_dim=input_dim,out_dim=self.action_dim,hidden_dim=self.hidden_dim,n_hidden=self.n_hidden,sigma_data=self.sigma_data).to(self.device)
        else:
            raise  ValueError("\n Model type should be either 'mlp' or  'resmlp' \n")
            

    def c_skip_fn(self,sigma, sigma_data):
        return sigma_data**2 / (sigma**2 + sigma_data**2)

    def c_out_fn(self,sigma, sigma_data):
        return sigma * sigma_data / torch.sqrt(sigma_data**2 + sigma**2)

    def c_in_fn(self,sigma, sigma_data):
        return 1.0 / torch.sqrt(sigma**2 + sigma_data**2)

    def c_noise_fn(self,sigma):
        return torch.log(sigma)*0.25

    def denoiser_fn(self, x, sigma, condition):
            x_scaled = self.c_in_fn(sigma, self.model.sigma_data) * x
            c_noise_expanded = self.c_noise_fn(sigma).expand(-1, 1)
            out = self.model(x_scaled,c_noise_expanded, condition)
            return self.c_skip_fn(sigma, self.model.sigma_data) * x + self.c_out_fn(sigma, self.model.sigma_data) * out
  
    def inference(self,state):
        
        batch_size = state.shape[0]
        
        sigmas = self.get_noise_schedule(self.n_diffusion_steps,self.noise_type).to(self.device)
        
        noise_action = sigmas[0] * torch.randn(batch_size, self.action_dim, device=self.device)
        
        action = self.sample_ddim(state,noise_action,sigmas)

        return action
        
    
    @torch.no_grad()
    def sample_ddim(
        self,
        state, 
        action, 
        sigmas, 
    ):
        """
        DPM-Solver 1( or DDIM sampler"""

        s_in = action.new_ones([action.shape[0]])
        sigma_fn = lambda t: t.neg().exp()
        t_fn = lambda sigma: sigma.log().neg()
        
  

        
        for i in trange(len(sigmas) - 1, disable=True):
            # predict the next action
            sigma_batch = (sigmas[i] * s_in).unsqueeze(1) 
            denoised = self.denoiser_fn(action, sigma_batch, state)
            t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
            h = t_next - t
            action = (sigma_fn(t_next) / sigma_fn(t)) * action - (-h).expm1() * denoised
            
        return action


            
        
    def get_noise_schedule(self,n_sampling_steps, noise_schedule_type):
        """
        Get the noise schedule for the sampling steps
        """
        if noise_schedule_type == 'karras':
            return get_sigmas_karras(n_sampling_steps, self.sigma_min, self.sigma_max, self.rho, self.device)
        elif noise_schedule_type == 'exponential':
            return get_sigmas_exponential(n_sampling_steps, self.sigma_min, self.sigma_max, self.device)
        elif noise_schedule_type == 'vp':
            return get_sigmas_vp(n_sampling_steps, device=self.device)
        elif noise_schedule_type == 'linear':
            return get_sigmas_linear(n_sampling_steps, self.sigma_min, self.sigma_max, device=self.device)
        elif noise_schedule_type == 'cosine_beta':
            return cosine_beta_schedule(n_sampling_steps, device=self.device)
        elif noise_schedule_type == 've':
            return get_sigmas_ve(n_sampling_steps, self.sigma_min, self.sigma_max, device=self.device)
        elif noise_schedule_type == 'iddpm':
            return get_iddpm_sigmas(n_sampling_steps, self.sigma_min, self.sigma_max, device=self.device)
        raise ValueError('Unknown noise schedule type')

    def old_sample_ddim(self, state, eta=0, num_sample=None):
        """
        DDIM sampling that generates either a single sample or multiple samples per state.

        Returns tensor of shape [num_sample, batch_size, action_dim] if num_sample is given,
        otherwise tensor of shape [batch_size, action_dim].
        """
        # Setup sampling parameters
        max_sigma = 80.0
        min_sigma = 1e-4
        sigmas = torch.exp(torch.linspace(
            torch.log(torch.tensor(max_sigma)),
            torch.log(torch.tensor(min_sigma)),
            self.n_diffusion_steps + 1
        )).to(self.device)

        batch_size = state.shape[0]

        if num_sample is not None:
            # For multiple samples, generate noise of shape [num_sample, batch_size, action_dim]
            # This ensures the output will be [num_sample, batch_size, action_dim]
            z = max_sigma * torch.randn(num_sample, batch_size, self.action_dim, device=self.device)

            # We need to process all samples together, so reshape state and noise for batch processing
            # Reshape state to [num_sample*batch_size, state_dim]
            state_flat = state.unsqueeze(0).expand(num_sample, -1, -1).reshape(-1, state.shape[-1])

            # Reshape noise to [num_sample*batch_size, action_dim]
            x_flat = z.reshape(-1, self.action_dim)

            # DDIM sampling loop for flattened tensors
            for i in range(self.n_diffusion_steps):
                sigma_i = sigmas[i]
                sigma_next = sigmas[i + 1]

                # Create properly shaped sigma tensor
                sigma_flat = torch.ones((x_flat.shape[0], 1), device=self.device) * sigma_i

                # Predict denoised sample
                with torch.no_grad():
                    denoised_flat = self.denoiser_fn(x_flat, sigma_flat, state_flat)

                # DDIM update formula
                x0_pred = denoised_flat
                dir_x0 = (x_flat - x0_pred) / sigma_i

                # Apply step
                if eta > 0:
                    # Stochastic DDIM
                    noise = torch.randn_like(x_flat)
                    sigma_t = eta * (sigma_next**2 / sigma_i**2).sqrt() * (1 - (sigma_next**2 / sigma_i**2)).sqrt()
                    x_flat = x0_pred + dir_x0 * sigma_next + noise * sigma_t
                else:
                    # Deterministic DDIM
                    x_flat = x0_pred + dir_x0 * sigma_next

            # Reshape back to [num_sample, batch_size, action_dim]
            x = x_flat.reshape(num_sample, batch_size, self.action_dim)

        else:
            # Single sample case - original implementation
            z = max_sigma * torch.randn(batch_size, self.action_dim, device=self.device)
            x = z

            for i in range(self.n_diffusion_steps):
                sigma_i = sigmas[i]
                sigma_next = sigmas[i + 1]

                sigma = torch.ones((batch_size, 1), device=self.device) * sigma_i

                with torch.no_grad():
                    denoised = self.denoiser_fn(x, sigma, state)

                x0_pred = denoised
                dir_x0 = (x - x0_pred) / sigma_i

                if eta > 0:
                    noise = torch.randn_like(x)
                    sigma_t = eta * (sigma_next**2 / sigma_i**2).sqrt() * (1 - (sigma_next**2 / sigma_i**2)).sqrt()
                    x = x0_pred + dir_x0 * sigma_next + noise * sigma_t
                else:
                    x = x0_pred + dir_x0 * sigma_next

        return x

   
    def forward(self, state, num_sample : int = 1):

        if num_sample>1:
            state_rep = state.unsqueeze(1).expand(-1, num_sample, -1).reshape(-1, state.size(-1)) 
        else:
            state_rep = state
              
            
        if self.sampler_type == 'ddim':
            action = self.inference(state_rep)

        action = action.reshape(state.shape[0], num_sample, self.action_dim).permute(1, 0, 2)
        
        if num_sample == 1:
            action = action.squeeze(0)

        return action.contiguous()


class Actor(torch.nn.Module):
    def __init__(self, encoder, torso, head):
        super().__init__()
        self.encoder = encoder
        self.torso = torso
        self.head = head

    def initialize(
        self, observation_space, action_space, observation_normalizer=None
    ):
        size = self.encoder.initialize(
            observation_space)
        size = self.torso.initialize(size)
        action_size = action_space.shape[0]
        self.head.initialize(size, action_size)

    def forward(self, *inputs):

        out = self.encoder(*inputs)
        out = self.torso(out)
        return self.head(out)

class DiffusionActor(torch.nn.Module):
    def __init__(self, encoder, torso, head):
        super().__init__()
        self.encoder = encoder
        self.torso = torso
        self.head = head

    def initialize(
        self, observation_space, action_space, observation_normalizer=None,actor_squash =False,actor_scale=1,
    ):
        size = self.encoder.initialize(
            observation_space)
        size = self.torso.initialize(size)
        action_size = action_space.shape[0]
        self.head.initialize(size, action_size)
        self.actor_squash = actor_squash
        self.actor_scale  = actor_scale
    def forward(self, *inputs):
        
        out = self.encoder(*inputs)
        out = self.torso(out)
        
        if len(out) > 1 and isinstance(out, tuple):
            out,samples = out
            pred = self.head(out,samples)
        else:
            pred = self.head(out)
           
        return pred



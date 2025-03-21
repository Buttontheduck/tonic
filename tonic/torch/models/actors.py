import torch
import torch.nn as nn
import torch.nn.functional as F
from tonic.torch.agents.diffusion_utils.diffusion_agents.k_diffusion.gc_sampling import *
from tonic.torch.agents.diffusion_utils.diffusion_agents.k_diffusion.score_wrappers import GCDenoiser  as trainer
from tonic.torch.agents.diffusion_utils.scaler import *
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
    def __init__(self, device="cpu", num_diffusion_steps=50, hidden_dim=256,n_hidden=4,n_blocks=6, sigma_data=1.0,sampler_type='ddim',model_type = 'mlp'):
        
        self.device = device
        self.hidden_dim = hidden_dim
        self.n_hidden=n_hidden
        self.sigma_data=sigma_data
        self.sampler_type=sampler_type
        self.n_blocks = n_blocks
        self.n_diffusion_steps = num_diffusion_steps
        self.model_type = model_type


    def initialize(self, state_size, action_size):
        super(DiffusionPolicyHead, self).__init__()
        self.state_dim = state_size
        self.action_dim = action_size
        input_dim = self.state_dim + self.action_dim +1
        
        
        if self.model_type=='mlp':
            self.model = ConditionalMLP(in_dim=input_dim, out_dim=self.action_dim,hidden_dim=self.hidden_dim,n_hidden=self.n_hidden,sigma_data=self.sigma_data).to(self.device)
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
        return torch.log(sigma)




    def denoiser_fn(self, x, sigma, condition):
        x_scaled = self.c_in_fn(sigma, self.model.sigma_data) * x
        c_noise_expanded = self.c_noise_fn(sigma).expand(-1, 1)
        inp = torch.cat([x_scaled, c_noise_expanded], dim=-1)
        out = self.model(inp, condition)
        return self.c_skip_fn(sigma, self.model.sigma_data) * x + self.c_out_fn(sigma, self.model.sigma_data) * out

    def velocity(self, x, t, condition):
        d = self.denoiser_fn( x, t, condition)
        return (x - d) / t

    # Modified ODE-based sampling method with condition
    def sample_ode(self, state,  atol=1e-5, rtol=1e-5):
        max_t = 80.0
        min_t = 1e-4


        batch_size = state.shape[0]
        z = max_t * torch.randn(batch_size, 2, device=self.device)
        shape = z.shape

        def velocity_wrapper(x_arr, time_steps):
            x_torch = torch.from_numpy(x_arr.astype(np.float32)).reshape(shape).to(self.device)
            ts_torch = torch.from_numpy(time_steps.astype(np.float32)).unsqueeze(-1).to(self.device)
            with torch.no_grad():
                v = self.velocity(self.model, x_torch, ts_torch, state)
            return v.view(-1).cpu().numpy().astype(np.float32)

        def ode_func(t_, x_arr):
            time_steps = np.ones((shape[0],), dtype=np.float32) * t_
            return velocity_wrapper(x_arr, time_steps)

        z_numpy = z.detach().cpu().numpy().reshape(-1).astype(np.float32)
        res = scipy.integrate.solve_ivp(
            ode_func, (max_t, min_t), z_numpy,
            rtol=rtol, atol=atol, method='RK45'
        )
        print(f"Number of function evaluations: {res.nfev}")
        final_result = res.y[:, -1].astype(np.float32)
        action_final = torch.from_numpy(final_result).to(self.device).reshape(shape)
        return action_final

    
    def sample_ddim(self, state, eta=0, num_sample=None):
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
    

    def forward(self, state, num_sample=None):
        """
        Forward pass to generate action samples given states.
        Args:
            state: Tensor of shape [batch_size, state_dim]
            num_sample: Optional int, number of action samples to generate per state

        Returns:
            If num_sample is None: Tensor of shape [batch_size, action_dim]
            If num_sample is given: Tensor of shape [num_sample, batch_size, action_dim]
        """
        if num_sample is None:
            # Single sample case
            if self.sampler_type == 'ddim':
                action = self.sample_ddim(state)
            elif self.sampler_type == 'ode':
                action = self.sample_ode(state)
        else:
            # Multiple samples case - vectorized
            if self.sampler_type == 'ddim':
                action = self.sample_ddim(state, num_sample=num_sample)
            elif self.sampler_type == 'ode':
                action = self.sample_ode(state, num_sample=num_sample)

        return action



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
        samples =None        
        out = self.encoder(*inputs)
        out = self.torso(out)
        if len(out) > 1 and isinstance(out, tuple):
            out,samples = out
            
        pred = self.head(out,samples)
        
        if self.actor_squash:
            pred = torch.tanh(pred) * self.actor_scale
            
        return pred



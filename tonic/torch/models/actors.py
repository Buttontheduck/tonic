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
    def __init__(self, device="cpu", num_diffusion_steps=50, hidden_dim=256,embed_dim=64,embed_type ='sinusoidal',n_hidden=4, sigma_data=1.0,sampler_type='ddim',model_type = 'mlp', \
        sigma_max=80, sigma_min=0.001, rho=7,  s_churn=0., s_tmin=0.,  s_tmax=float('inf'), s_noise=1.,eta = 1.0, noise_type = 'karras' ):
        
        self.device = device
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.embed_type = embed_type
        self.n_hidden=n_hidden
        self.sigma_data=sigma_data
        self.sampler_type=sampler_type
        self.n_diffusion_steps = num_diffusion_steps
        self.model_type = model_type
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.rho = rho
        self.s_churn = s_churn
        self.s_tmin = s_tmin
        self.s_tmax = s_tmax
        self.s_noise = s_noise
        self.noise_type = noise_type
        self.eta = eta
        


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
        return torch.log(sigma)*1

    def denoiser_fn(self, noised_action, sigma, state):
            scaled_noised_action= self.c_in_fn(sigma, self.model.sigma_data) * noised_action
            c_noise_expanded = self.c_noise_fn(sigma).expand(-1, 1)
            inner_model_output = self.model(scaled_noised_action, c_noise_expanded, state)
            return self.c_skip_fn(sigma, self.model.sigma_data) * noised_action + self.c_out_fn(sigma, self.model.sigma_data) * inner_model_output
  
    def inference(self,state):
        
        batch_size = state.shape[0]
        sigmas = self.get_noise_schedule(self.n_diffusion_steps,self.noise_type).to(self.device)
        noised_action = sigmas[0] * torch.randn(batch_size, self.action_dim, device=self.device)
        
        if self.sampler_type =='ddim':
            action = self.sample_ddim(state, noised_action, sigmas)
            
        elif self.sampler_type =='heun':
            action = self.sample_heun(state, noised_action ,sigmas, s_churn=self.s_churn, \
                                      s_tmin = self.s_tmin, s_tmax = self.s_tmax, s_noise = self.s_noise)
            
        elif self.sampler_type == 'dpm':
            action = self.sample_dpm_2(state, noised_action ,sigmas, s_churn=self.s_churn, \
                                      s_tmin = self.s_tmin, s_tmax = self.s_tmax, s_noise = self.s_noise)
            
        elif self.sampler_type == 'dpm_ancestral':
            action = self.sample_dpm_2_ancestral(state, noised_action,  sigmas, self.eta)
            
        elif self.sampler_type =='euler':
            action = self.sample_euler(state, noised_action ,sigmas, s_churn=self.s_churn, \
                                      s_tmin = self.s_tmin, s_tmax = self.s_tmax, s_noise = self.s_noise)
            
        elif self.sampler_type == 'euler_ancestral':
            action = self.sample_euler_ancestral(state, noised_action,  sigmas, self.eta)
            
        else:
            raise ValueError("\n  Sampler type is not assigned correctly, Choose  'ddim' , 'heun' , 'euler' , 'euler_ancestral' , 'dpm' , 'dpm_ancestral' \n")

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
    
    @torch.no_grad()
    def sample_heun(
        self,
        state, 
        action, 
        sigmas, 
        s_churn=0., 
        s_tmin=0., 
        s_tmax=float('inf'), 
        s_noise=1.
    ):
        """
        Implements Algorithm 2 (Heun steps) from Karras et al. (2022).
        For S_churn =0 this is an ODE solver otherwise SDE
        Every update consists of these substeps:
        1. Addition of noise given the factor eps
        2. Solving the ODE dx/dt at timestep t using the score model 
        3. Take Euler step from t -> t+1 to get x_{i+1}
        4. 2nd order correction step to get x_{i+1}^{(2)}

        In contrast to the Euler variant, this variant computes a 2nd order correction step. 
        """
        

        s_in = action.new_ones([action.shape[0]])
        for i in trange(len(sigmas) - 1, disable=True):
            
           
            sigma_batch_next  = (sigmas[i + 1] * s_in).unsqueeze(1)
            
            gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
            eps = torch.randn_like(action) * s_noise
            sigma_hat = sigmas[i] * (gamma + 1)
            # if gamma > 0, use additional noise level for computation ODE-> SDE Solver
            if gamma > 0:
                action= action+ eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
            denoised = self.denoiser_fn(action, (sigma_hat * s_in).unsqueeze(1), state)
            d = to_d(action, sigma_hat, denoised)
            dt = sigmas[i + 1] - sigma_hat
            # if we only are at the last step we use an Euler step for our update otherwise the heun one 
            if sigmas[i + 1] == 0:
                # Euler method
                action = action + d * dt
            else:
                # Heun's method
                action_2 = action + d * dt
                denoised_2 = self.denoiser_fn(action_2,sigma_batch_next,state)
                d_2 = to_d( action_2, sigmas[i + 1], denoised_2)
                d_prime = (d + d_2) / 2
                action= action+ d_prime * dt

        return action
    
    @torch.no_grad()
    def sample_dpm_2(
        self, 
        state, 
        action, 
        sigmas, 
        s_churn=0., 
        s_tmin=0., 
        s_tmax=float('inf'), 
        s_noise=1.
    ):
        """
        A sampler inspired by DPM-Solver-2 and Algorithm 2 from Karras et al. (2022).
        SDE for S_churn!=0 and ODE otherwise

        1.

        Last denoising step is an Euler step  
        """

        s_in = action.new_ones([action.shape[0]])
        for i in trange(len(sigmas) - 1, disable=True):
            # compute stochastic gamma if s_churn > 0: 
            gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.

            eps = torch.randn_like(action) * s_noise
            sigma_hat = sigmas[i] * (gamma + 1)
            # add noise to our current action sample in SDE case
            if gamma > 0:
                action = action + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
            # compute the derivative dx/dt at timestep t
            denoised = self.denoiser_fn(action, (sigma_hat * s_in).unsqueeze(1), state)
            d = to_d(action, sigma_hat, denoised)

            # if we are at the last timestep: use Euler method
            if sigmas[i + 1] == 0:
                # Euler method
                dt = sigmas[i + 1] - sigma_hat
                action = action + d * dt
            else:
                # use Heun 2nd order update step 
                sigma_mid = sigma_hat.log().lerp(sigmas[i + 1].log(), 0.5).exp()
                dt_1 = sigma_mid - sigma_hat
                dt_2 = sigmas[i + 1] - sigma_hat
                action_2 = action + d * dt_1
                denoised_2 = self.denoiser_fn(action_2, (sigma_mid * s_in).unsqueeze(1), state)
                d_2 = to_d( action_2, sigma_mid, denoised_2)
                action = action + d_2 * dt_2
        return action
    
    @torch.no_grad()
    def sample_dpm_2_ancestral(self, state, action,  sigmas, eta=1.):
        """
        Ancestral sampling with DPM-Solver inspired second-order steps.

        Ancestral sampling is based on the DDPM paper (https://arxiv.org/abs/2006.11239) generation process.
        Song et al. (2021) show that ancestral sampling can be used to improve the performance of DDPM for its SDE formulation.

        1. Compute dx_{i}/dt at the current timestep 

        """

        s_in = action.new_ones([action.shape[0]])
        for i in trange(len(sigmas) - 1, disable=True):
           
            denoised = self.denoiser_fn(action, (sigmas[i] * s_in).unsqueeze(1),state)
            sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
            d = to_d(action, sigmas[i], denoised)
            if sigma_down == 0:
                # Euler method
                dt = sigma_down - sigmas[i]
                action= action+ d * dt
            else:
                # DPM-Solver-2
                sigma_mid = sigmas[i].log().lerp(sigma_down.log(), 0.5).exp()
                dt_1 = sigma_mid - sigmas[i]
                dt_2 = sigma_down - sigmas[i]
                action_2 = action+ d * dt_1
                denoised_2 = self.denoiser_fn(action_2, (sigma_mid * s_in).unsqueeze(1), state)
                d_2 = to_d( action_2, sigma_mid, denoised_2)
                action= action + d_2 * dt_2
                action= action + torch.randn_like(action) * sigma_up

        return action      

    @torch.no_grad()
    def sample_euler(
        self, 
        state: torch.Tensor, 
        action: torch.Tensor,  
        sigmas, 
        s_churn=0., 
        s_tmin=0., 
        s_tmax=float('inf'), 
        s_noise=1.
    ):
        """
        Implements a variant of Algorithm 2 (Euler steps) from Karras et al. (2022).
        Stochastic sampler, which combines a first order ODE solver with explicit Langevin-like "churn"
        of adding and removing noise. 
        Every update consists of these substeps:
        1. Addition of noise given the factor eps
        2. Solving the ODE dx/dt at timestep t using the score model 
        3. Take Euler step from t -> t+1 to get x_{i+1}

        In contrast to the Heun variant, this variant does not compute a 2nd order correction step
        For S_churn=0 the solver is an ODE solver
        """

        s_in = action.new_ones([action.shape[0]])
        for i in trange(len(sigmas) - 1, disable=True):
            gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0. 
            eps = torch.randn_like(action) * s_noise    # sample current noise depnding on S_noise 
            sigma_hat = sigmas[i] * (gamma + 1)         # add noise to sigma
            # print(action[:, -1, :])
            if gamma > 0: # if gamma > 0, use additional noise level for computation
                action = action + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5 
            denoised = self.denoiser_fn(action, (sigma_hat * s_in).unsqueeze(1), state) # compute denoised action
            d = to_d(action, sigma_hat, denoised) # compute derivative
            dt = sigmas[i + 1] - sigma_hat # compute timestep
            # Euler method
            action = action + d * dt # take Euler step
        return action
 

    @torch.no_grad()
    def sample_euler_ancestral(
        self, 
        state, 
        action, 
        sigmas,
        eta=1.
        ):
        """
        Ancestral sampling with Euler method steps.

        1. compute dx_{i}/dt at the current timestep 
        2. get \sigma_{up} and \sigma_{down} from ancestral method 
        3. compute x_{t-1} = x_{t} + dx_{t}/dt * \sigma_{down}
        4. Add additional noise after the update step x_{t-1} =x_{t-1} + z * \sigma_{up}
        """
        s_in = action.new_ones([action.shape[0]])
        for i in trange(len(sigmas) - 1, disable=True):
            # compute x_{t-1}
            denoised = self.denoiser_fn(action, (sigmas[i] * s_in).unsqueeze(1), state)
            # get ancestral steps
            sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
            # compute dx/dt 
            d = to_d(action, sigmas[i], denoised)
            # compute dt based on sigma_down value 
            dt = sigma_down - sigmas[i]
            # update current action 
            action = action + d * dt
            if sigma_down > 0:
                action = action + torch.randn_like(action) * sigma_up
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


    def forward(self, state, num_sample : int = 1):

        if num_sample>1:
            state_rep = state.unsqueeze(1).expand(-1, num_sample, -1).reshape(-1, state.size(-1)) 
        else:
            state_rep = state
            
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
        self, observation_space, action_space, observation_normalizer=None):
        size = self.encoder.initialize(
            observation_space)
        size = self.torso.initialize(size)
        action_size = action_space.shape[0]
        self.head.initialize(size, action_size)

    def forward(self, *inputs):
        
        out = self.encoder(*inputs)
        out = self.torso(out)
        
        if len(out) > 1 and isinstance(out, tuple):
            out,samples = out
            pred = self.head(out,samples)
        else:
            pred = self.head(out)
           
        return pred


    """  
    def old_sample_ddim(self, state, eta=0, num_sample=None):
        
        #DDIM sampling that generates either a single sample or multiple samples per state.

        #Returns tensor of shape [num_sample, batch_size, action_dim] if num_sample is given,
        #otherwise tensor of shape [batch_size, action_dim].
    
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
        """

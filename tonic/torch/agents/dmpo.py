import torch

from tonic import logger, replays  # noqa
from tonic.torch import agents, models, normalizers, updaters

from tonic.torch.agents.diffusion_utils.utils import IdentityEncoder, IdentityTorso


def default_model():
    return models.DiffusionActorCriticWithTargets(
        actor=models.DiffusionActor(
            encoder=IdentityEncoder(),
            torso=IdentityTorso(),
            head=models.DiffusionPolicyHead(device='cpu',num_diffusion_steps=50, hidden_dim=100,n_hidden=2,sampler_type='ddim',model_type='mlp',sigma_data=1)),
        critic=models.Critic(
            encoder=models.ObservationActionEncoder(),
            torso=models.MLP((256, 256), torch.nn.ReLU),
            head=models.ValueHead()),
        observation_normalizer=normalizers.MeanStd(),
        actor_squash=False,
        action_scale=1)


class DMPO(agents.Agent):
    

    def __init__(
        self, model, replay, actor_updater, critic_updater
    ):
        self.model = build_model(model) or default_model() 
        self.replay =  replays.Buffer(return_steps=5)
        self.actor_updater = build_actor_updater(actor_updater) or \
            updaters.DiffusionMaximumAPosterioriPolicyOptimization()
        self.critic_updater = build_critic_updater(critic_updater) or updaters.DiffusionExpectedSARSA()
        
        

    def initialize(self, observation_space, action_space, seed=None):
        super().initialize(seed=seed)
        self.model.initialize(observation_space, action_space)
        self.replay.initialize(seed)
        self.actor_updater.initialize(self.model, action_space)
        self.critic_updater.initialize(self.model)

    def step(self, observations, steps):
        actions = self._step(observations)
        actions = actions.numpy()

        # Keep some values for the next update.
        self.last_observations = observations.copy()
        self.last_actions = actions.copy()

        return actions

    def test_step(self, observations, steps):
        # Sample actions for testing.
        return self._test_step(observations).numpy()

    def update(self, observations, rewards, resets, terminations, steps):
        # Store the last transitions in the replay.
        self.replay.store(
            observations=self.last_observations, actions=self.last_actions,
            next_observations=observations, rewards=rewards, resets=resets,
            terminations=terminations)

        # Prepare to update the normalizers.
        if self.model.observation_normalizer:
            self.model.observation_normalizer.record(self.last_observations)
        if self.model.return_normalizer:
            self.model.return_normalizer.record(rewards)

        # Update the model if the replay is ready.
        if self.replay.ready(steps):
            self._update(steps)

    def _step(self, observations):
        observations = torch.as_tensor(observations, dtype=torch.float32)
        with torch.no_grad():
            return self.model.actor(observations)

    def _test_step(self, observations):
        observations = torch.as_tensor(observations, dtype=torch.float32)
        with torch.no_grad():
            return self.model.actor(observations)

    def _update(self, steps):
        keys = ('observations', 'actions', 'next_observations', 'rewards',
                'discounts')

        # Update both the actor and the critic multiple times.
        for batch in self.replay.get(*keys, steps=steps):
            batch = {k: torch.as_tensor(v) for k, v in batch.items()}
            infos = self._update_actor_critic(**batch)

            for key in infos:
                for k, v in infos[key].items():
                    logger.store(key + '/' + k, v.numpy())

        # Update the normalizers.
        if self.model.observation_normalizer:
            self.model.observation_normalizer.update()
        if self.model.return_normalizer:
            self.model.return_normalizer.update()
            


    def _update_actor_critic(
        self, observations, actions, next_observations, rewards, discounts
    ):
        critic_infos = self.critic_updater(
            observations, actions, next_observations, rewards, discounts)
        actor_infos = self.actor_updater(observations) 
        self.model.update_targets()
        return dict(critic=critic_infos, actor=actor_infos)
    
    
    
    
    
    
    
    
def build_model(cfg):

    head_cfg = cfg["actor"]["head"]
    actor_head = models.DiffusionPolicyHead(
        device=head_cfg["device"],
        num_diffusion_steps=head_cfg["num_diffusion_steps"],
        hidden_dim=head_cfg["hidden_dim"],
        n_hidden=head_cfg["n_hidden"],
        n_blocks=head_cfg["n_blocks"],
        sigma_data=head_cfg["sigma_data"],
        sampler_type=head_cfg["sampler_type"],
        model_type=head_cfg["model_type"]
    )


    if cfg["actor"]["encoder"]["name"] == "IdentityEncoder":
        actor_encoder = IdentityEncoder()


    if cfg["actor"]["torso"]["name"] == "IdentityTorso":
        actor_torso = IdentityTorso()


    actor = models.DiffusionActor(
        encoder=actor_encoder,
        torso=actor_torso,
        head=actor_head
    )

    # 3) Build critic
    critic_cfg = cfg["critic"]
    # Encoder
    if critic_cfg["encoder"]["name"] == "ObservationActionEncoder":
        critic_encoder = models.ObservationActionEncoder()

    # Torso
    torso_cfg = critic_cfg["torso"]
    if torso_cfg["name"] == "MLP":
        hidden_layers = tuple(torso_cfg["hidden_layers"])
        activation = getattr(torch.nn, torso_cfg["activation"])
        critic_torso = models.MLP(hidden_layers, activation)

    # Head
    if critic_cfg["head"]["name"] == "ValueHead":
        critic_head = models.ValueHead()

    critic = models.Critic(
        encoder=critic_encoder,
        torso=critic_torso,
        head=critic_head
    )

    # 4) Observation normalizer
    obs_norm_cfg = cfg["observation_normalizer"]
    if obs_norm_cfg and obs_norm_cfg["name"] == "MeanStd":
        observation_normalizer = normalizers.MeanStd()
    else:
        observation_normalizer = None


    return models.DiffusionActorCriticWithTargets(
        actor=actor,
        critic=critic,
        observation_normalizer=observation_normalizer,
        actor_squash=cfg["actor_squash"],
        action_scale=cfg["action_scale"],
        target_coeff=cfg.get("target_coeff", 0.005)
    )


def build_actor_updater(cfg):
   
    actor_cfg = cfg
    
    if actor_cfg["name"] == "DiffusionMaximumAPosterioriPolicyOptimization":

        optim_cfg = actor_cfg["optimizer"]
        if optim_cfg["name"] == "Adam":
            learning_rate = optim_cfg["learning_rate"]
            actor_optimizer = lambda params: torch.optim.Adam(params, lr=learning_rate)

            dual_optimizer = None

        

        return updaters.DiffusionMaximumAPosterioriPolicyOptimization(
            num_samples=actor_cfg["num_samples"],
            epsilon=actor_cfg["epsilon"],
            epsilon_penalty=actor_cfg["epsilon_penalty"],
            epsilon_mean=actor_cfg["epsilon_mean"],
            epsilon_std=actor_cfg["epsilon_std"],
            initial_log_temperature=actor_cfg["initial_log_temperature"],
            initial_log_alpha_mean=actor_cfg["initial_log_alpha_mean"],
            initial_log_alpha_std=actor_cfg["initial_log_alpha_std"],
            min_log_dual=actor_cfg["min_log_dual"],
            per_dim_constraining=actor_cfg["per_dim_constraining"],
            action_penalization=actor_cfg["action_penalization"],
            actor_optimizer=actor_optimizer,
            dual_optimizer=dual_optimizer,
            gradient_clip=actor_cfg["gradient_clip"]
        )
    
    raise ValueError(f"Unsupported actor updater: {actor_cfg['name']}")


def build_critic_updater(cfg):

    critic_cfg = cfg
    
    if critic_cfg["name"] == "DiffusionExpectedSARSA":
        # Get optimizer
        optim_cfg = critic_cfg["optimizer"]
        if optim_cfg["name"] == "Adam":
            learning_rate = optim_cfg["learning_rate"]
            critic_optimizer = lambda params: torch.optim.Adam(params, lr=learning_rate)
        # Add more optimizers as needed
        
        # Create the updater
        return updaters.DiffusionExpectedSARSA(
            num_samples=critic_cfg["num_samples"],
            optimizer=critic_optimizer,
            gradient_clip=critic_cfg["gradient_clip"]
        )
    
    raise ValueError(f"Unsupported critic updater: {critic_cfg['name']}")


from .foodcollector.foodcollector_base import FoodCollector as _env
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from pettingzoo.utils import wrappers
import numpy as np
from gym import spaces
from collections import defaultdict
import copy
from pettingzoo.utils.env import ParallelEnv
# from ..utils.conversions import to_parallel_wrapper


def env(**kwargs):
    env = raw_env(**kwargs)
    if not kwargs["dist_action"]:
        env = wrappers.ClipOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env

class to_parallel_wrapper_comm(ParallelEnv):
    def __init__(self, aec_env):
        self.aec_env = aec_env
        self.observation_spaces = aec_env.observation_spaces
        self.action_spaces = aec_env.action_spaces
        self.communication_spaces = aec_env.communication_spaces
        self.possible_agents = aec_env.possible_agents
        self.metadata = aec_env.metadata
        # Not every environment has the .state_space attribute implemented
        try:
            self.state_space = self.aec_env.state_space
        except AttributeError:
            pass

    @property
    def unwrapped(self):
        return self.aec_env.unwrapped

    def seed(self, seed=None):
        return self.aec_env.seed(seed)

    def reset(self):
        self.aec_env.reset()
        self.agents = self.aec_env.agents
        observations = {agent: self.aec_env.observe(agent) for agent in self.aec_env.agents if not self.aec_env.dones[agent]}
        communications = {agent: self.aec_env.communicate(agent) for agent in self.aec_env.agents}
        return observations, communications

    def step(self, actions):
        while self.aec_env.agents and self.aec_env.dones[self.aec_env.agent_selection]:
            self.aec_env.step(None)

        rewards = {a: 0 for a in self.aec_env.agents}
        dones = {}
        infos = {}
        observations = {}

        for agent in self.aec_env.agents:
            assert agent == self.aec_env.agent_selection, f"expected agent {agent} got agent {self.aec_env.agent_selection}, agent order is nontrivial"
            # obs, rew, done, info = self.aec_env.last()
            self.aec_env.step(actions[agent])
            for agent in self.aec_env.agents:
                rewards[agent] += self.aec_env.rewards[agent]

        dones = dict(**self.aec_env.dones)
        infos = dict(**self.aec_env.infos)
        self.agents = self.aec_env.agents
        observations = {agent: self.aec_env.observe(agent) for agent in self.aec_env.agents}
        communications = {agent: self.aec_env.communicate(agent) for agent in self.aec_env.agents}
        return observations, communications, rewards, dones, infos

    def render(self, mode="human"):
        return self.aec_env.render(mode)

    def state(self):
        return self.aec_env.state()

    def close(self):
        return self.aec_env.close()

def parallel_comm_wrapper_fn(env_fn):
    def par_fn(**kwargs):
        print(kwargs)
        env = env_fn(**kwargs)
        env = to_parallel_wrapper_comm(env)
        return env
    return par_fn

parallel_env = parallel_comm_wrapper_fn(env)

class raw_env(AECEnv):

    metadata = {'render.modes': ['human', "rgb_array"], 'name': 'foodcollector_v1'}

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.env = _env(comm=True, *args, **kwargs)

        self.agents = ["pursuer_" + str(r) for r in range(self.env.num_agents)]
        self.possible_agents = self.agents[:]
        self.agent_name_mapping = dict(zip(self.agents, list(range(self.num_agents))))
        self._agent_selector = agent_selector(self.agents)
        # spaces
        self.action_spaces = dict(zip(self.agents, self.env.action_space))
        self.observation_spaces = dict(
            zip(self.agents, self.env.observation_space))
        self.communication_spaces = dict(
            zip(self.agents, self.env.communication_space))
        print(self.communication_spaces)
        self.has_reset = False

    def seed(self, seed=None):
        self.env.seed(seed)

    def convert_to_dict(self, list_of_list):
        return dict(zip(self.agents, list_of_list))

    def reset(self):
        self.has_reset = True
        self.env.reset()
        self.agents = self.possible_agents[:]
        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.next()
        self.rewards = dict(zip(self.agents, [(0) for _ in self.agents]))
        self._cumulative_rewards = dict(zip(self.agents, [(0) for _ in self.agents]))
        self.dones = dict(zip(self.agents, [False for _ in self.agents]))
        self.infos = dict(zip(self.agents, [{} for _ in self.agents]))

    def close(self):
        if self.has_reset:
            self.env.close()

    def render(self, mode="human"):
        return self.env.render(mode)

    def step(self, action):
        if self.dones[self.agent_selection]:
            return self._was_done_step(action)
        agent = self.agent_selection

        is_last = self._agent_selector.is_last()
        self.env.step(action, self.agent_name_mapping[agent], is_last)

        for r in self.rewards:
            self.rewards[r] = self.env.control_rewards[self.agent_name_mapping[r]]
        if is_last:
            for r in self.rewards:
                self.rewards[r] += self.env.last_rewards[self.agent_name_mapping[r]]

        if self.env.frames >= self.env.max_cycles:
            self.dones = dict(zip(self.agents, [True for _ in self.agents]))
        else:
            self.dones = dict(zip(self.agents, self.env.last_dones))
        self._cumulative_rewards[self.agent_selection] = 0
        self.agent_selection = self._agent_selector.next()
        self._accumulate_rewards()

    def observe(self, agent):
        return self.env.observe(self.agent_name_mapping[agent])
    
    def communicate(self, agent):
        return self.env.communicate(self.agent_name_mapping[agent])


class to_adversary_wrapper_comm(ParallelEnv):
    def __init__(self, aec_env, adv_names, victim_name, train_victim):
        """
        aec_env: the base environment
        adv_names: all attackers
        victim_name: victim
        train_victim: whether it is robust training. 
            If True: fix the attacker policy while training the victim
            If False: fix the victim policy while training the attacker
        """
        self.aec_env = aec_env
        self.adv_names = adv_names
        self.good_names = []
        self.default_policy = None
        self.adv_default_policy = None
        self.victim = victim_name
        self.train_victim = train_victim
        ## initializing agents and spaces
        self.possible_agents = aec_env.possible_agents
        self.adv_observation_spaces, self.adv_action_spaces, self.adv_communication_spaces = {}, {}, {}
        self.good_observation_spaces, self.good_action_spaces, self.good_communication_spaces = {}, {}, {}
        adv_comm_dim = aec_env.communication_spaces[self.victim].shape[0]//(len(self.possible_agents)-1)
        for agent in self.possible_agents:
            if agent in adv_names:
                self.adv_observation_spaces[agent] = aec_env.observation_spaces[agent]
                self.adv_action_spaces[agent] = spaces.Box(low=-1., high=1., shape=(adv_comm_dim,), dtype=np.float32)
                # self.action_spaces[agent] = spaces.Box(low=-np.inf, 
                #     high=np.inf, shape=(adv_comm_dim,), dtype=np.float32)
                self.adv_communication_spaces[agent] = aec_env.communication_spaces[agent]
            else:
                self.good_names.append(agent)
            self.good_observation_spaces[agent] = aec_env.observation_spaces[agent]
            self.good_action_spaces[agent] = aec_env.action_spaces[agent]
            self.good_communication_spaces[agent] = aec_env.communication_spaces[agent]
        print("normal obs space", self.good_observation_spaces)
        print("normal action space", self.good_action_spaces)
        print("communication space", self.good_communication_spaces)
        print("adv obs space", self.adv_observation_spaces)
        print("adv action space", self.adv_action_spaces)
        
        good_comm_dim = self.good_communication_spaces[self.victim].shape[0]
        single_comm_channel = good_comm_dim // (len(self.possible_agents)-1)
        self.attack_dim_start, self.attack_dim_end = {}, {}
        for adv in self.adv_names:
            idx = 0
            for agent in self.possible_agents:
                if agent == adv:
                    break
                if agent != self.victim:
                    idx += 1
                
            self.attack_dim_start[adv] = (idx) * single_comm_channel 
            self.attack_dim_end[adv] = (idx+1) * single_comm_channel
        print("attack at index", self.attack_dim_start, self.attack_dim_end)

        ### different default spaces with different trainable agents
        if self.train_victim:
            self.observation_spaces = self.good_observation_spaces
            self.communication_spaces = self.good_communication_spaces
            self.action_spaces = self.good_action_spaces
        else:
            self.observation_spaces = self.adv_observation_spaces
            self.communication_spaces = self.adv_communication_spaces
            self.action_spaces = self.adv_action_spaces

        self.all_rewards = defaultdict(int)

        self.metadata = aec_env.metadata
        # Not every environment has the .state_space attribute implemented
        try:
            self.state_space = self.aec_env.state_space
        except AttributeError:
            pass

    def set_default_policy(self, policy):
        self.default_policy = policy
    
    def set_adv_default_policy(self, policy):
        self.adv_default_policy = policy
    
    def set_victim(self, victim):
        self.victim = victim

    @property
    def unwrapped(self):
        return self.aec_env.unwrapped

    def seed(self, seed=None):
        return self.aec_env.seed(seed)

    def reset(self):
        self.aec_env.reset()
        self.agents = self.aec_env.agents
        observations = {agent: self.aec_env.observe(agent) for agent in self.aec_env.agents if not self.aec_env.dones[agent]}
        communications = {agent: self.aec_env.communicate(agent) for agent in self.aec_env.agents}
        self.last_observations = copy.deepcopy(observations)
        self.last_communications = copy.deepcopy(communications)
        self.all_rewards = defaultdict(int)
        return observations, communications

    def step(self, actions):
        while self.aec_env.agents and self.aec_env.dones[self.aec_env.agent_selection]:
            self.aec_env.step(None)

        rewards = {a: 0 for a in self.aec_env.agents}
        dones = {}
        infos = {}
        observations = {}

        # if training the adversary, send the adversarial comm
        if not self.train_victim:
            for adv in self.adv_names:
                ### send bad communications
                self.last_communications[self.victim][self.attack_dim_start[adv]:self.attack_dim_end[adv]] = actions[adv]
        
        for agent in self.aec_env.agents:
            ### take moving actions
            assert agent == self.aec_env.agent_selection, f"expected agent {agent} got agent {self.aec_env.agent_selection}, agent order is nontrivial"
            if self.train_victim:
                normal_action = actions[agent]
            else:
                normal_action = self.default_policy.act(self.last_observations[agent], self.last_communications[agent])
            self.aec_env.step(normal_action)
            
            for agent in self.aec_env.agents:
                if not self.train_victim and agent in self.adv_names:
                    rewards[agent] -= self.aec_env.rewards[self.victim]
                else:
                    rewards[agent] += self.aec_env.rewards[agent]
                self.all_rewards[agent] += self.aec_env.rewards[agent]

        dones = dict(**self.aec_env.dones)
        infos = dict(**self.aec_env.infos)
        self.agents = self.aec_env.agents
        observations = {agent: self.aec_env.observe(agent) for agent in self.aec_env.agents}
        communications = {agent: self.aec_env.communicate(agent) for agent in self.aec_env.agents}
        # if training victim, use the default adv policy to perturb the victim comm
        if self.train_victim:
            for adv in self.adv_names:
                adv_action = self.adv_default_policy.act(self.last_observations[adv], self.last_communications[adv])
                communications[self.victim][self.attack_dim_start[adv]:self.attack_dim_end[adv]] = adv_action
        self.last_observations = copy.deepcopy(observations)
        self.last_communications = copy.deepcopy(communications)
        return observations, communications, rewards, dones, infos

    def render(self, mode="human"):
        return self.aec_env.render(mode)

    def state(self):
        return self.aec_env.state()

    def close(self):
        return self.aec_env.close()

def parallel_advcomm_wrapper_fn(env_fn, adv_names, victim, train_victim=False):
    def par_fn(**kwargs):
        env = env_fn(**kwargs)
        env = to_adversary_wrapper_comm(env, adv_names, victim, train_victim)
        return env
    return par_fn

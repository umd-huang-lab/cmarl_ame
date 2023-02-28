from ._mpe_utils.simple_env import SimpleEnv, make_env
from .scenarios.simple_exchange import Scenario
from pettingzoo.utils.conversions import parallel_wrapper_fn
from ._mpe_utils.core import ShareStrategy
from gym import spaces
import numpy as np

class raw_env(SimpleEnv):
    def __init__(self, N=3, local_ratio=0.5, max_cycles=25, beta=1):
        assert 0. <= local_ratio <= 1., "local_ratio is a proportion. Must be between 0 and 1."
        scenario = Scenario()
        world = scenario.make_world(N, beta=beta, comm_type="say")
        super().__init__(scenario, world, max_cycles, local_ratio)
        self.metadata['name'] = "simple_exchange_v1"
        self.N = N

        # set spaces
        self.action_spaces = dict()
        self.observation_spaces = dict()
        state_dim = 0
        for agent in self.world.agents:
            space_dim = 1
            if agent.movable:
                space_dim *= self.world.dim_p * 2 + 1
            if not agent.silent:
                space_dim *= self.world.dim_c
            if agent.protective:
                space_dim *= self.world.dim_share ** (N-1)

            obs_dim = len(self.scenario.observation(agent, self.world))
            state_dim += obs_dim
            self.action_spaces[agent.name] = spaces.Discrete(space_dim)
            self.observation_spaces[agent.name] = spaces.Box(low=-np.float32(np.inf), high=+np.float32(np.inf), shape=(obs_dim,), dtype=np.float32)

        self.state_space = spaces.Box(low=-np.float32(np.inf), high=+np.float32(np.inf), shape=(state_dim,), dtype=np.float32)

    def convert_agent(self, agent_name):
        for agent in self.world.agents:
            if agent.name == agent_name:
                agent.adversary = True
                print("converting", agent.name, agent.adversary)
                
    def reset_local(self, ratio):
        self.local_ratio = ratio
    
    def disable_adv_reward(self):
        self.scenario.alpha = 0
        self.scenario.beta = 1

    def _execute_world_step(self):
        # set action for each agent
        for i, agent in enumerate(self.world.agents):
            action = self.current_actions[i]
            scenario_action = []
            if agent.movable:
                mdim = self.world.dim_p * 2 + 1
                scenario_action.append(action % mdim)
                action //= mdim
            if not agent.silent:
                scenario_action.append(action % self.world.dim_c)
                action //= self.world.dim_c
            if agent.protective:
                for i in range(self.N-1):
                    scenario_action.append(action % self.world.dim_share)
                    action //= self.world.dim_share
            self._set_action(scenario_action, agent, self.action_spaces[agent.name])

        self.world.step()

        global_reward = 0.
        if self.local_ratio is not None:
            global_reward = float(self.scenario.global_reward(self.world))

        for agent in self.world.agents:
            agent_reward = float(self.scenario.reward(agent, self.world))
            if self.local_ratio is not None:
                reward = global_reward * (1 - self.local_ratio) + agent_reward * self.local_ratio
            else:
                reward = agent_reward

            self.rewards[agent.name] = reward

    # set env action for a particular agent
    def _set_action(self, action, agent, action_space, time=None):
        agent.action.u = np.zeros(self.world.dim_p)
        agent.action.c = np.zeros(self.world.dim_c)
        # process action
        if agent.movable:
            # physical action
            agent.action.u = np.zeros(self.world.dim_p)
            # process discrete action
            if action[0] == 1:
                agent.action.u[0] = -1.0
            if action[0] == 2:
                agent.action.u[0] = +1.0
            if action[0] == 3:
                agent.action.u[1] = -1.0
            if action[0] == 4:
                agent.action.u[1] = +1.

            sensitivity = 5.0
            if agent.accel is not None:
                sensitivity = agent.accel
            agent.action.u *= sensitivity
            action = action[1:]
        if not agent.silent:
            # communication action
            agent.action.c = np.zeros(self.world.dim_c)

            agent.action.c[action[0]] = 1.0
            action = action[1:]
        if agent.protective:
            for other in self.world.agents:
                if other is agent:
                    continue
                else:
                    if action[0] == 0:
                        agent.share_strategy[other.name] = ShareStrategy.ALWAYS
                    elif action[0] == 1:
                        agent.share_strategy[other.name] = ShareStrategy.NO
                    elif action[0] == 2:
                        agent.share_strategy[other.name] = ShareStrategy.NOISY
                    action = action[1:]
            # print(agent.share_strategy)
        # make sure we used all elements of action
        assert len(action) == 0
    
    def step(self, action):
        if self.dones[self.agent_selection]:
            return self._was_done_step(action)
        cur_agent = self.agent_selection
        current_idx = self._index_map[self.agent_selection]
        next_idx = (current_idx + 1) % self.num_agents
        self.agent_selection = self._agent_selector.next()

        self.current_actions[current_idx] = action

        if next_idx == 0:
            self._execute_world_step()
            self.steps += 1
            if self.steps >= self.max_cycles:
                for a in self.agents:
                    self.dones[a] = True
        else:
            self._clear_rewards()

        self._cumulative_rewards[cur_agent] = 0
        self._accumulate_rewards()

env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)

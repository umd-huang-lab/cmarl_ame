from ._mpe_utils.simple_env import SimpleEnv, make_env
from .scenarios.simple_adversary_info import Scenario
from pettingzoo.utils.conversions import parallel_wrapper_fn
from gym import spaces
import numpy as np

class raw_env(SimpleEnv):
    def __init__(self, N=2, max_cycles=25, share="policy"):
        scenario = Scenario()
        world = scenario.make_world(N, share=share)
        super().__init__(scenario, world, max_cycles)
        self.metadata['name'] = "simple_adversary_v3"
        
        # reset action spaces
        self.action_spaces = dict()
        for agent in self.world.agents:
            space_dim = 1
            if agent.movable:
                space_dim *= self.world.dim_p * 2 + 1
            if not agent.silent:
                space_dim *= self.world.dim_c
            if agent.protective:
                space_dim *= 2 # hide pos, reveal pos
            self.action_spaces[agent.name] = spaces.Discrete(space_dim)
        
        # self.action_space = self.action_spaces[self.world.agents[0].name]
        # self.enabled = True
    
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
                scenario_action.append(action)

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
            # mask agent
            if action[0] == 0:
                agent.is_masked = True
            elif action[0] == 1:
                agent.is_masked = False
            action = action[1:]
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

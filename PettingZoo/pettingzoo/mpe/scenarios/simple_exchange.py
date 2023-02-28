import numpy as np
from numpy.lib.type_check import common_type
from .._mpe_utils.core import World, Agent, Landmark, ShareStrategy
from .._mpe_utils.scenario import BaseScenario 

class Scenario(BaseScenario):
    def make_world(self, N=3, beta=1, comm_type="pos", victim_id=0, landmark_movable=False):
        '''
        comm_type: the type of communication. ["pos": communicate positions, "say": send signals] 
        '''
        world = World()
        self.N = N
        self.beta = beta
        self.alpha = 1
        self.comm_type = comm_type
        print(self.comm_type)
        # set any world properties first
        world.dim_c = 10
        world.dim_share = 3
        world.collaborative = True  # whether agents share rewards

        colors = [
            np.array([0.75, 0.25, 0.25]),
            np.array([0.25, 0.75, 0.25]),
            np.array([0.25, 0.25, 0.75]),
        ]

        # add agents
        world.agents = [Agent() for i in range(N)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent_{}'.format(i)
            agent.collide = True
            if self.comm_type == "pos":
                agent.silent = True
                agent.protective = True
            agent.color = colors[i] 
        # add landmarks
        world.landmarks = [Landmark() for i in range(N)]
        self.landmark_movable = landmark_movable
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = landmark_movable
            landmark.color = colors[i]
        # assign goals and observations
        for i, agent in enumerate(world.agents):
            agent.observed_lms = []
            for j, landmark in enumerate(world.landmarks):
                if i == j:
                    agent.goal_a = landmark
                else:
                    agent.observed_lms.append(landmark)
        
        for agent in world.agents:
            print("agent", agent.name, "has goal", agent.goal_a.name)
            print("agent", agent.name, "observes")
            for l in agent.observed_lms:
                print(l.name)

        world.victim = world.agents[victim_id]

        return world

    def reset_world(self, world, np_random):
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for landmark in world.landmarks:
            landmark.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def reward(self, agent, world):
        agent_reward = - np.sqrt(np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos)))
        if not agent.adversary:
            return agent_reward
        else:
            victim = world.victim
            assert victim is not agent, "adversarial agent and victim should not be the same"
            adv_reward = np.sqrt(np.sum(np.square(victim.state.p_pos - victim.goal_a.state.p_pos)))
            return self.alpha * adv_reward + self.beta * agent_reward

    def global_reward(self, world):
        all_rewards, num_agents = 0, 0
        for agent in world.agents:
            if not agent.adversary:
                all_rewards += self.reward(agent, world)
                num_agents  += 1
        return all_rewards / num_agents
    
    def info_share(self, from_agent, to_agent, world):
        assert from_agent != to_agent, "information sharing can only happen between different agents."
        # print(from_agent.name, "to", to_agent.name, "strategy:", from_agent.share_strategy[to_agent.name])
        if from_agent.share_strategy[to_agent.name] == ShareStrategy.NO:
            return [0.0] * world.dim_p * (1 + len(from_agent.observed_lms))
        else:
            ### Info includess the relative position between two agents
            info = [from_agent.state.p_pos - to_agent.state.p_pos]
            for lm in from_agent.observed_lms:
                info.append(lm.state.p_pos - from_agent.state.p_pos)
            if from_agent.share_strategy[to_agent.name] == ShareStrategy.NOISY:
                for k in info:
                    k += np.random.randn(world.dim_p)*from_agent.share_noise
            return np.concatenate(info)
                
    def observation(self, agent, world):
        # landmark positions
        lm_pos = []
        for lm in agent.observed_lms:
            lm_pos.append(lm.state.p_pos)

        # communication of all other agents
        comm = []
        for other in world.agents:
            if other is agent:
                continue
            if not other.silent:
                comm.append(other.state.c)
            else:
                comm.append(self.info_share(other, agent, world))
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + lm_pos + comm)

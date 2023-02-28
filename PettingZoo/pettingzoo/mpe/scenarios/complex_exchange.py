import numpy as np
from numpy.lib.type_check import common_type
from .._mpe_utils.core import World, Agent, Landmark, ShareStrategy
from .._mpe_utils.scenario import BaseScenario
from .._mpe_utils.message_center import MessageCenter

class Scenario(BaseScenario):
    def make_world(self, N=3, beta=1, victim_id=0):
        '''
        make a complex information exchange environment
        '''
        world = World()
        self.N = N
        self.beta = beta
        self.alpha = 1
        
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
            agent.id = i
            agent.name = 'agent_{}'.format(i)
            agent.collide = True
            agent.silent = True
            agent.protective = True
            agent.color = colors[i] 
        # add landmarks
        world.landmarks = [Landmark() for i in range(N)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.color = colors[i]
        
        world.victim = world.agents[victim_id]

        world.msg_center = MessageCenter()

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
        
        # re-assign goals and observations
        # ensure that the goal and the observation does not conflict
        goal_order = np.arange(self.N)
        np_random.shuffle(goal_order)
        observe_shift = np_random.choice(np.arange(1,self.N))
        observe_order = np.concatenate((goal_order[observe_shift:], goal_order[:observe_shift]))

        for i, agent in enumerate(world.agents):
            agent.goal_a = world.landmarks[goal_order[i]]
            agent.goal_id = goal_order[i]
            agent.observed_lm = world.landmarks[observe_order[i]]
            agent.observed_id = observe_order[i]
        
        # for agent in world.agents:
        #     print("agent", agent.name, "has goal", agent.goal_a.name)
        #     print("agent", agent.name, "observes", agent.observed_lm.name)
        world.msg_center.reset()

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
        all_rewards = sum(self.reward(agent, world) for agent in world.agents)
        return all_rewards / len(world.agents)
    
    def info_share(self, from_agent, to_agent, world):
        assert from_agent != to_agent, "information sharing can only happen between different agents."
        info = from_agent.observed_lm.state.p_pos
        return info
    
    def observe_communication(self, agent, world):
        msg = world.msg_center.get_message(agent.id)
        # if msg:
        #     print("agent", agent.name, "receives a message from", msg.sender_id)
        return world.msg_center.decode_message(msg, agent.goal_id, self.N, world.dim_p)
                
    def observation(self, agent, world):
        comm = self.observe_communication(agent, world)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + \
            [agent.observed_lm.state.p_pos] + [comm])

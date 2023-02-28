from collections import defaultdict
import torch
import matplotlib
from torch.optim import Adam
from torch.nn import GRU, Linear
import torch
import matplotlib
from torch.optim import Adam
import gym
import time
from copy import deepcopy
import itertools
from core import *
from param import Param
import pickle
import concurrent.futures as futures
from pettingzoo.sisl import foodcollector_v0, foodcollector_v1
#from ppo import DefaultPolicy
        
### Default Policy for benign agents
class DefaultPolicy:
    def __init__(self, policy_net, dist_action=False, 
                       ablate_kwargs=None, confidence_kwargs=None,
                       detector_ac=None) -> None:
        self.policy_net    = policy_net.to(Param.device)
        self.dist_action   = dist_action
        self.ablate_kwargs = ablate_kwargs
        self.idx_list      = np.array(makeCombi(ablate_kwargs['n']-1, ablate_kwargs['k']))
        if confidence_kwargs is None:
            self.detector_ac = None
            self.act_bias = None
            self.step = None
            self.filter_count = None
        else:
            self.detector_ac = detector_ac
            self.act_bias = None
            self.step = 0
            ### Count the number of times that each agent has been filtered out
            self.filter_count = dict([(i,0) for i in range(self.ablate_kwargs['n'])])
            
    def act(self, observation, communication):
        ### Randomized Ablation
        if self.ablate_kwargs is not None:
            k,n = self.ablate_kwargs['k'], self.ablate_kwargs['n']
            comm = communication.reshape(n-1, -1)
            if self.act_bias is None:
                choice = np.random.choice(self.idx_list.shape[0], self.ablate_kwargs['median'], 
                                          replace=False)
                idxs = [self.idx_list[i] for i in choice]
                o = np.stack([observation]*self.ablate_kwargs['median'])
                comm = np.concatenate([comm[idx].reshape(1,-1) for idx in idxs])
            else:
                idx = np.argsort(self.act_bias)[:k]
                o, comm = observation, comm[idx].reshape(-1)
                ### Update the filter count list
                filter_idx = np.argsort(self.act_bias)[k:]
                for idx in filter_idx:
                    self.filter_count[idx] += 1
                
        else:
            o, comm = observation, communication
        o = np.concatenate([o, comm],axis=-1)
        a, _, _ = self.policy_net.step(torch.from_numpy(o).\
                                to(Param.device).type(Param.dtype), train=False)
        #a = torch.tensor([a[idxs.index(i)] for i in range(8)],torch.int32)[2:].to(Param.device)
        
        if len(a.shape) > 1 and not self.dist_action:
            a = torch.median(a, axis=0)[0]
        elif self.dist_action:
            a = torch.mode(a, axis=-1)[0]
        if self.dist_action:
            a = a.item()
        else:
            a = a.cpu().numpy()
    
        ### If confidence_kwargs is not None, update the confidence score
        if self.ablate_kwargs is not None and self.detector_ac is not None:
            self.update_act_bias(observation, communication)
        return a
    
    def update_act_bias(self, observation, communication):
        
        ### Concatenate observation with each of the agent's communication
        n = self.ablate_kwargs['n']
        o = np.stack([observation]*(n-1))
        communication = communication.reshape(n-1, -1)
        o = np.concatenate([o, communication],axis=-1)
        
        ### Compute Action based on each agent
        actions, _, _ = self.detector_ac.step(torch.from_numpy(o).\
                                to(Param.device).type(Param.dtype), train=False)
        ### Compute Median Action
        median_a = torch.median(actions,axis=0, keepdim=True)[0]
        ### Compute Action Bias
        act_bias = torch.sum(torch.abs(actions - median_a),axis=-1).cpu().numpy()
        
        self.step += 1
        if self.act_bias is None:
            self.act_bias = act_bias
        else:
            self.act_bias = self.act_bias*(self.step-1)/self.step + act_bias/self.step 
    
def shuffle(comm, k):
    choice = np.random.choice(comm.shape[0], k, replace=False)   
    return comm[choice]



def make_advcomm_env(adv_agents, good_policy_dir, victim, ac_kwargs, ablate_kwargs, 
                     confidence_kwargs, **kwargs):
    env_fn = foodcollector_v1.parallel_advcomm_wrapper_fn(foodcollector_v1.env, 
        adv_agents, victim)
    env = env_fn(**kwargs)

    a = env.good_names[0]
    good_observation_space = env.good_observation_spaces[a]
    good_comm_space = env.good_communication_spaces[a]
    good_action_space = env.good_action_spaces[a]
    if ablate_kwargs is None:
        obs_dim = good_observation_space.shape[0] + good_comm_space.shape[0]
    else:
        k, n = ablate_kwargs['k'], ablate_kwargs['n']
        obs_dim =  good_observation_space.shape[0] + good_comm_space.shape[0]*k//(n-1)
    if isinstance(good_action_space, Box):
        ac_kwargs['beta'] = True
    else:
        ac_kwargs['beta'] = False
    ac = MLPActorCritic(obs_dim, good_action_space, **ac_kwargs).to(Param.device)
    if good_policy_dir:
        state_dict, mean, std = torch.load(good_policy_dir, map_location=Param.device)
        ac.load_state_dict(state_dict)
        ac.moving_mean = mean
        ac.moving_std = std
    
    detector_ac = None
    if confidence_kwargs is not None:
        k, n = 1, ablate_kwargs['n']
        obs_dim =  good_observation_space.shape[0] + good_comm_space.shape[0]*k//(n-1)
        detector_ac = MLPActorCritic(obs_dim, good_action_space, **ac_kwargs).to(Param.device)
        state_dict, mean, std = torch.load(confidence_kwargs['detector_policy_dir'], map_location=Param.device)
        detector_ac.load_state_dict(state_dict)
        detector_ac.moving_mean = mean
        detector_ac.moving_std = std
        
    default_policy = DefaultPolicy(ac, True if kwargs["victim_dist_action"] else False, 
                                   ablate_kwargs=ablate_kwargs, confidence_kwargs=confidence_kwargs,
                                   detector_ac=detector_ac)
    env.set_default_policy(default_policy)
    env.reset()
    return env, adv_agents, []

                          
def make_food_env(comm, **kwargs):
    if comm:
        env = foodcollector_v1.parallel_env(**kwargs)
    else:
        env = foodcollector_v0.parallel_env(**kwargs)
    env.reset()

    env.reset()
    agent = env.agents[0]
    observation_space = env.observation_spaces[agent]
    action_space = env.action_spaces[agent]
    
    adv_agents = []
    good_agents = ["pursuer_{}".format(i) for i in range(kwargs["n_pursuers"])]
    #for a in all_agents:
        #if convert and a in convert:
            #adv_agents.append(a)
            #env.unwrapped.convert_agent(a)
        #else:
    #good_agents.append(a)
    return env, good_agents, adv_agents

def main(env_fn=None, actor_critic=MLPActorCritic, ac_kwargs=dict(), trained_dir=None, 
        beta=False, comm=False, dist_action=False, render=False, 
        recurrent=False, adv_train=False, ablate_kwargs=None, count_agent=None, 
        max_ep_len=200, epochs=20, num_agent=9, num_adv=0, 
        random_attacker=False, test_random = False
        ):

    
    print('-----Test Performance {} Epochs---'.format(epochs))
    
    # Instantiate environment
    env, good_agent_name, adv_agent_name = env_fn()
    observation_space = env.observation_spaces[good_agent_name[0]]
    action_space      = env.action_spaces[good_agent_name[0]]
    act_dim           = action_space.shape
    if comm:
        comm_space    = env.communication_spaces[good_agent_name[0]]
        if ablate_kwargs is None or ablate_kwargs['adv_train']:
            obs_dim       = (observation_space.shape[0]  + comm_space.shape[0],)
        else:
            num_agents = len(good_agent_name) + len(adv_agent_name)
            obs_dim       = (observation_space.shape[0]  + 
                             comm_space.shape[0]*ablate_kwargs['k']//(num_agents-1),)
    else:
        obs_dim = env.observation_spaces[good_agent_name[0]].shape 
    if beta:
        high = torch.from_numpy(action_space.high).to(Param.device).type(Param.dtype)
        low = torch.from_numpy(action_space.low).to(Param.device).type(Param.dtype)
    #print("Obs dim:{}, Act dim:{}".format(obs_dim[0], act_dim))
    
    # Create actor-critic module
    ac = actor_critic(obs_dim[0], action_space, **ac_kwargs).to(Param.device)
    if trained_dir is not None:
        state_dict, mean, std = torch.load(trained_dir, map_location=Param.device)
        ac.load_state_dict(state_dict)
        ac.moving_mean = mean
        ac.moving_std = std
    
    if random_attacker:
        ac = RandomAttacker(action_space)
    
    mean, _ = test_return(env, ac, epochs, max_ep_len, good_agent_name, 
                            dist_action, comm, recurrent, ablate_kwargs,
                            random_ac = test_random)
    print('-----[Number of Agents:{} Number of Adversary:{} Ablation Size:{} Median Sample Size {}] {} Epochs Performance:{}------'.\
          format(num_agent, num_adv, ablate_kwargs['k'], ablate_kwargs['median'], epochs, mean))
   

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action="store_true")
    parser.add_argument('--cuda', type=int, default=2)
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--beta', action="store_true")
    parser.add_argument('--trained-dir', type=str)
    parser.add_argument('--render', action="store_true")
    parser.add_argument('--smart', action="store_true")
    
    ### environment setting
    parser.add_argument('--window-size', type=int, default=1)
    parser.add_argument('--n-pursuers', type=int, default=3)
    parser.add_argument('--n-evaders', type=int, default=1)
    parser.add_argument('--n-poison', type=int, default=1)
    parser.add_argument('--poison-scale', type=float, default=0.75)
    parser.add_argument('--n-sensors',  type=int, default=6)
    parser.add_argument('--max-cycle', type=int, default=200)
    parser.add_argument('--comm', action="store_true")
    parser.add_argument('--comm-freq',  type=int, default=1)
    parser.add_argument('--dist-action', action="store_true")
    parser.add_argument('--victim-dist-action', action="store_true")
    parser.add_argument('--sensor-range',  type=float, default=0.2)
    parser.add_argument('--evader-speed', type=float, default=0)
    parser.add_argument('--poison-speed', type=float, default=0)
    parser.add_argument('--speed-features', action="store_true")
    parser.add_argument('--recurrent', action="store_true")
    parser.add_argument('--food-revive', action="store_true", help="whether the food can be refreshed after being eaten")
    parser.add_argument('--truth', action="store_true")
    parser.add_argument('--obs-normalize', action="store_true")
    
    parser.add_argument('--test-random-attacker', action="store_true")
    parser.add_argument('--convert-adv', type=str, nargs='+')
    parser.add_argument('--good-policy', type=str)
    parser.add_argument('--victim', type=str, default="pursuer_0")
    
    parser.add_argument('--ablate', action='store_true')
    parser.add_argument('--ablate-k', type=int, default=1)
    parser.add_argument('--ablate-median', type=int, default=1)
    parser.add_argument('--test-random', action="store_true")
    
    parser.add_argument('--alpha', type=float, default = 0.0, 
                        help='exponential moving average for confidence reweighting ablation')
    parser.add_argument('--detector-policy-dir', type=str, default=None)
    args = parser.parse_args()
    gym.logger.set_level(40)
    
    if args.ablate:
        ablate_kwargs={"k":args.ablate_k, "n":args.n_pursuers, 
                       "adv_train": True if args.convert_adv else False, 
                       "adv_agents": args.convert_adv if args.convert_adv else [],
                       "median":args.ablate_median}
    else:
        ablate_kwargs=None
    
    confidence_kwargs = {'alpha':args.alpha,
                         'detector_policy_dir':args.detector_policy_dir} if args.alpha > 0 else None
    
    if args.no_cuda:
        Param(torch.FloatTensor, torch.device("cpu"))
    else:
        Param(torch.cuda.FloatTensor, torch.device("cuda:{}".format(args.cuda)))
    
    ### Please make sure the following argument names are the same as the FoodCollector's init function
    if args.convert_adv:
        env = make_advcomm_env(adv_agents=args.convert_adv, good_policy_dir=args.good_policy, victim=args.victim,
                ac_kwargs=dict(hidden_sizes=[args.hid]*args.l, 
                            recurrent=args.recurrent, ep_len=args.max_cycle),
                ablate_kwargs = ablate_kwargs, confidence_kwargs = confidence_kwargs,
                window_size=args.window_size, poison_scale=args.poison_scale,  food_revive=args.food_revive,
                max_cycles=args.max_cycle, n_pursuers=args.n_pursuers, n_evaders=args.n_evaders, 
                n_poison=args.n_poison, n_sensors=args.n_sensors, dist_action=args.victim_dist_action,
                sensor_range=args.sensor_range, evader_speed=args.evader_speed, poison_speed=args.poison_speed,
                speed_features=args.speed_features, use_groudtruth=args.truth, smart_comm=args.smart, 
                comm_freq=args.comm_freq, victim_dist_action=args.victim_dist_action)
    else:
        env = make_food_env(
                comm=args.comm, max_cycles=args.max_cycle,
                window_size=args.window_size, poison_scale=args.poison_scale, food_revive=args.food_revive,
                n_pursuers=args.n_pursuers, n_evaders=args.n_evaders, 
                n_poison=args.n_poison, n_sensors=args.n_sensors, dist_action=args.dist_action,
                sensor_range=args.sensor_range, evader_speed=args.evader_speed, poison_speed=args.poison_speed,
                speed_features=args.speed_features, use_groudtruth=args.truth, 
                smart_comm=args.smart, comm_freq=args.comm_freq)
        
    main(lambda:env, actor_critic=MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l, beta=args.beta, 
                       recurrent=args.recurrent, ep_len=args.max_cycle), 
        epochs=args.epochs, max_ep_len=args.max_cycle,
        beta=args.beta, comm=args.comm, dist_action=args.dist_action, trained_dir=args.trained_dir, 
        render=args.render, recurrent=args.recurrent, 
        adv_train=True if args.convert_adv else False,
        ablate_kwargs=ablate_kwargs, num_agent=args.n_pursuers, 
        num_adv = len(args.convert_adv) if args.convert_adv is not None else 0,
        random_attacker=args.test_random_attacker, test_random = args.test_random)

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
from pettingzoo.sisl import foodcollector_v0, foodcollector_v1
from ppo import ppo

class DefaultPolicy:
    def __init__(self, policy_net, dist_action=False, adv=False, num_agents=9, shuffle=True) -> None:
        self.policy_net  = policy_net
        self.dist_action = dist_action
        self.n           = num_agents
        self.adv         = adv
        self.shuffle     = shuffle
        self.act_bias    = None
        self.detector_ac = None
        
    def act(self, observation, communication):
        ### If not adversary, only shuffle the agent
        if not self.adv and self.shuffle:
            communication = communication.reshape(self.n-1,-1)
            np.random.shuffle(communication)
            communication = np.concatenate(communication)
        o = np.concatenate([observation, communication])
        a, _, _ = self.policy_net.step(torch.from_numpy(o).\
                                to(Param.device).type(Param.dtype), train=False)
        if self.dist_action:
            a = a.item()
        else:
            a = a.cpu().numpy()
        return a

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
    return env, good_agents, adv_agents

def make_advcomm_env(train_victim, adv_agents, good_policy_dir, adv_policy_dir,
        victim, ac_kwargs, **kwargs):
    env_fn = foodcollector_v1.parallel_advcomm_wrapper_fn(foodcollector_v1.env, 
        adv_agents, victim, train_victim)
    env = env_fn(**kwargs)

    print("good agents", env.good_names)
    print("adv agents", env.adv_names)
    print("victim", env.victim)

    if train_victim:
        ac_kwargs['beta'] = False
        a = env.adv_names[0]
        adv_observation_space = env.adv_observation_spaces[a]
        adv_comm_space = env.adv_communication_spaces[a]
        adv_action_space = env.adv_action_spaces[a]
        obs_dim = adv_observation_space.shape[0] + adv_comm_space.shape[0]
        adv_ac = MLPActorCritic(obs_dim, adv_action_space, **ac_kwargs).to(Param.device)
        if adv_policy_dir:
            print("loaded pretrained adv model from", adv_policy_dir)
            state_dict, mean, std = torch.load(adv_policy_dir, map_location=Param.device)
            adv_ac.load_state_dict(state_dict)
            adv_ac.moving_mean = mean
            adv_ac.moving_std = std
        adv_default_policy = DefaultPolicy(adv_ac, False,
                                           adv=True, shuffle=False)
        env.set_adv_default_policy(adv_default_policy)
        env.reset()
        return env, env.agents, []
    else:
        ac_kwargs['beta'] = True if not kwargs["dist_action"] else False
        a = env.good_names[0]
        good_observation_space = env.good_observation_spaces[a]
        good_comm_space = env.good_communication_spaces[a]
        good_action_space = env.good_action_spaces[a]
        obs_dim = good_observation_space.shape[0] + good_comm_space.shape[0]
        print(ac_kwargs['beta'])
        print(good_action_space)
        good_ac = MLPActorCritic(obs_dim, good_action_space, **ac_kwargs).to(Param.device)
        if good_policy_dir:
            state_dict, mean, std = torch.load(good_policy_dir, map_location=Param.device)
            good_ac.load_state_dict(state_dict)
            good_ac.moving_mean = mean
            good_ac.moving_std = std
            good_ac.MovingMeanStd.old_m = mean
            good_ac.MovingMeanStd.old_s = std
            good_ac.MovingMeanStd.n     = 40000
            print("loaded pretrained good model from", good_policy_dir)
        num_agents = len(env.good_names) + len(env.adv_names)
        good_default_policy = DefaultPolicy(good_ac, True if kwargs["dist_action"] else False,
                                            num_agents=num_agents, adv=False, shuffle=shuffle)
        env.set_default_policy(good_default_policy)

        env.reset()
        return env, adv_agents, []

def test_good(env_func, victim):
    env, agents, _ = env_func()
    action_space = env.action_spaces[victim] 
    rewards = []
    for i in range(50):
        o, c = env.reset()
        done = False
        agent_rewards = 0
        t = 0
        while True:
            actions = {}
            for agent in agents:
                a = action_space.sample()
                actions[agent] = a
            next_o, c, reward, dones,_  = env.step(actions)
            agent_rewards += reward[victim]
            for j in range(len(agents)):
                agent = agents[j]
                if dones[agent]:
                    done = True
            o = next_o
            # print(i, t, reward)
            t += 1
            if done:
                rewards.append(agent_rewards)
                print(i, rewards[-1])
                break
    print("mean reward", np.mean(rewards)) 

def test_pretrain(env_func):
    env, agents, _ = env_func()
    action_space = env.action_spaces[agents[0]] 
    rewards = []
    for i in range(50):
        o, c = env.reset()
        done = False
        agent_rewards = np.zeros(len(agents))
        t = 0
        while True:
            actions = {}
            for agent in agents:
                a = action_space.sample()
                actions[agent] = a
            next_o, c, reward, dones,_  = env.step(actions)
            for j in range(len(agents)):
                agent = agents[j]
                if dones[agent]:
                    done = True
                agent_rewards[j] += reward[agent]
            o = next_o
            # print(i, t, reward)
            t += 1
            if done:
                rewards.append(np.mean(agent_rewards))
                print(i, rewards[-1])
                break
    print("mean reward", np.mean(rewards)) 

def test_adv(env_func):
    env, agents, _ = env_func()
    action_space = env.action_spaces[agents[0]] 
    rewards = []
    for i in range(50):
        o, c = env.reset()
        done = False
        agent_rewards = np.zeros(len(agents))
        t = 0
        while True:
            actions = {}
            for agent in agents:
                a = action_space.sample()
                actions[agent] = a
            next_o, c, reward, dones,_  = env.step(actions)
            for j in range(len(agents)):
                agent = agents[j]
                if dones[agent]:
                    done = True
                agent_rewards[j] += reward[agent]
            o = next_o
            # print(i, t, reward)
            t += 1
            if done:
                rewards.append(np.mean(agent_rewards))
                print(i, rewards[-1])
                break
    print("mean reward", np.mean(rewards)) 

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', action="store_true")
    parser.add_argument('--no-save', action="store_true")
    parser.add_argument('--no-cuda', action="store_true")
    parser.add_argument('--vf-lr', type=float, default=1e-3)
    parser.add_argument('--pi-lr', type=float, default=3e-4)
    parser.add_argument('--cuda', type=int, default=2)
    parser.add_argument('--env', type=str, default='FoodCollector')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--steps', type=int, default=5000)
    parser.add_argument('--epoch-smoothed', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--exp-name', type=str, default='ppo')
    parser.add_argument('--log-freq', type=int, default=50)
    parser.add_argument('--obs-normalize', action="store_true")
    parser.add_argument('--beta', action="store_true")
    parser.add_argument('--trained-dir', type=str)
    parser.add_argument('--render', action="store_true")
    parser.add_argument('--truth', action="store_true")
    parser.add_argument('--smart', action="store_true")
    
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
    parser.add_argument('--sensor-range',  type=float, default=0.2)
    parser.add_argument('--evader-speed', type=float, default=0)
    parser.add_argument('--poison-speed', type=float, default=0)
    parser.add_argument('--speed-features', action="store_true")
    parser.add_argument('--recurrent', action="store_true")
    parser.add_argument('--food-revive', action='store_true')
    
    # about adversarial training
    parser.add_argument('--train-victim', action="store_true")
    parser.add_argument('--convert-adv', type=str, nargs='+')
    parser.add_argument('--good-policy', type=str)
    parser.add_argument('--adv-policy', type=str)
    parser.add_argument('--victim', type=str, default="pursuer_0")
    parser.add_argument('--no-shuffle', action='store_true')
    
    args = parser.parse_args()
    gym.logger.set_level(40)
    
    if args.no_cuda:
        Param(torch.FloatTensor, torch.device("cpu"))
    else:
        Param(torch.cuda.FloatTensor, torch.device("cuda:{}".format(args.cuda)))

    if args.convert_adv:
        env = make_advcomm_env(train_victim=args.train_victim,
                adv_agents=args.convert_adv, victim=args.victim,
                good_policy_dir=args.good_policy, adv_policy_dir=args.adv_policy, 
                ac_kwargs=dict(hidden_sizes=[args.hid]*args.l, beta=False, 
                            recurrent=args.recurrent, ep_len=args.max_cycle),
                window_size=args.window_size, poison_scale=args.poison_scale,
                max_cycles=args.max_cycle, n_pursuers=args.n_pursuers, n_evaders=args.n_evaders, 
                n_poison=args.n_poison, n_sensors=args.n_sensors, dist_action=args.dist_action,
                sensor_range=args.sensor_range, evader_speed=args.evader_speed, poison_speed=args.poison_speed,
                speed_features=args.speed_features, use_groudtruth=args.truth, smart_comm=args.smart,
                comm_freq=args.comm_freq, food_revive=args.food_revive, shuffle=False)
    else:
        env = make_food_env(
                comm = args.comm, max_cycles=args.max_cycle,
                window_size=args.window_size, poison_scale=args.poison_scale,
                n_pursuers=args.n_pursuers, n_evaders=args.n_evaders, 
                n_poison=args.n_poison, n_sensors=args.n_sensors, dist_action=args.dist_action,
                sensor_range=args.sensor_range, evader_speed=args.evader_speed, 
                poison_speed=args.poison_speed,speed_features=args.speed_features, 
                use_groudtruth=args.truth, smart_comm=args.smart, 
                comm_freq=args.comm_freq, food_revive=args.food_revive, shuffle=not args.no_shuffle)
 
    logger_file = open(os.path.join(Param.data_dir, r"logger_{}.txt".format(args.exp_name)), "wt")
    logger_file.write("Number of Pursuers: {}  Number of Food:{}  Number of Poison:{}\n".\
                      format(args.n_pursuers, args.n_evaders, args.n_poison))
    logger_file.write("Number of Sensors: {}  Sensor Range:{}    Speed Features:{}\n".\
                      format(args.n_sensors, args.sensor_range, args.speed_features))
    logger_file.write("Food Speed: {}  Poison Speed:{}\n".\
                      format(args.evader_speed, args.poison_speed))
    logger_file.close()

    # if args.train_victim:
    #     test_good(lambda: env, args.victim)
    # elif args.convert_adv:
    #     test_adv(lambda: env)
    # else:
    #     test_pretrain(lambda: env)
    if args.train_victim:
        trained_dir = args.good_policy
    else:
        trained_dir = args.adv_policy
    
    if (args.train_victim or args.convert_adv is None) and not args.no_shuffle:
        ablate_kwargs={"k":args.n_pursuers-1, "n":args.n_pursuers, 
                       "adv_train": False, 
                       "adv_agents": []}
    else:
        ablate_kwargs=None
        
    ppo(lambda:env, actor_critic=MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l, beta=args.beta, 
                       recurrent=args.recurrent, ep_len=args.max_cycle), 
        gamma=args.gamma, vf_lr=args.vf_lr, pi_lr=args.pi_lr,
        seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs, max_ep_len=args.max_cycle,
        name_env = args.env, epoch_smoothed=args.epoch_smoothed,
        no_save = args.no_save, verbose=args.verbose, log_freq = args.log_freq, 
        exp_name=args.exp_name, obs_normalize = args.obs_normalize, beta=args.beta, comm=args.comm, 
        dist_action=True if args.dist_action and args.train_victim else False, 
        render=args.render, recurrent=args.recurrent, adv_train=True if args.convert_adv else False,
        trained_dir=trained_dir, ablate_kwargs=ablate_kwargs, 
        count_agent=args.victim)

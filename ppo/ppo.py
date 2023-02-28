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
import re

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

def concatenate(obs, comm, ablate_kwargs=None):
    for agent in obs:
        if ablate_kwargs is None or agent in ablate_kwargs["adv_agents"]:
            obs[agent] = np.concatenate([obs[agent], comm[agent]])
        else:
            comm_agent = comm[agent].reshape(len(obs)-1, -1)
            comm_agent = shuffle(comm_agent, ablate_kwargs['k']).reshape(-1)
            obs[agent] = np.concatenate([obs[agent], comm_agent])
    return obs

class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    obs_buf, act_buf, adv_buf, rew_buf, ret_buf, val_buf: numpy array
    logp_array: torch tensor
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf  = np.zeros(combined_shape(size, obs_dim))
        self.act_buf  = np.zeros(combined_shape(size, act_dim))
        self.adv_buf  = np.zeros(size)
        self.rew_buf  = np.zeros(size)
        self.ret_buf  = np.zeros(size)
        self.val_buf  = np.zeros(size)
        self.logp_buf = torch.zeros(size).to(Param.device).type(Param.dtype)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp, hiddens=None):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.
        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)
        
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr

    
    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = np.mean(self.adv_buf, 0), np.std(self.adv_buf, 0)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=from_numpy(self.obs_buf), act=from_numpy(self.act_buf), ret=from_numpy(self.ret_buf),
                    adv=from_numpy(self.adv_buf), logp=self.logp_buf)
        return data

def make_advcomm_env(adv_agents, good_policy_dir, victim, ac_kwargs, ablate_kwargs, 
                     confidence_kwargs, **kwargs):
    env_fn = foodcollector_v1.parallel_advcomm_wrapper_fn(foodcollector_v1.env, 
        adv_agents, victim)
    env = env_fn(**kwargs)

    #print("good agents", env.good_names)
    #print("adv agents", env.adv_names)
    #print("victim", env.victim)
    a = env.good_names[0]
    good_observation_space = env.good_observation_spaces[a]
    good_comm_space = env.good_communication_spaces[a]
    #print("shape of good comomunication space:{}".format(good_comm_space.shape))
    good_action_space = env.good_action_spaces[a]
    if ablate_kwargs is None:
        obs_dim = good_observation_space.shape[0] + good_comm_space.shape[0]
    else:
        k, n = ablate_kwargs['k'], ablate_kwargs['n']
        obs_dim =  good_observation_space.shape[0] + good_comm_space.shape[0]*k//(n-1)
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
# def make_food_env(comm, n_pursuers, n_evaders, n_poison, n_sensors=12, 
#                   max_cycles=500, convert=[], dist_action=False, 
#                   sensor_range=0.2, evader_speed=0.005,poison_speed=0.005,
#                   speed_features=False, use_groudtruth=False, smart_comm=False, 
#                   comm_freq=1):
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

def ppo(env_fn=None, actor_critic=MLPActorCritic, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=4000, epochs=50, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
        vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=1000,
        target_kl=0.01, save_freq=10, name_env="HalfCheetah-v3", epoch_smoothed=10,
        no_save=False, verbose=False, log_freq=50, exp_name='ppo', trained_dir=None, 
        obs_normalize=False, beta=False, comm=False, dist_action=False, render=False, 
        recurrent=False, adv_train=False, ablate_kwargs=None, count_agent='pursuer_0'):
 
    # Setup logger file and reward file
    logger_file = open(os.path.join(Param.data_dir, r"logger_{}.txt".format(exp_name)), "a")
    rew_file = open(os.path.join(Param.data_dir, r"reward_{}.txt".format(exp_name)), "wt")
    
    # Random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Instantiate environment
    env, good_agent_name, adv_agent_name = env_fn()
    print("good agent:{}".format(good_agent_name))
    print("adv_agent_name:{}".format(adv_agent_name))
    
    observation_space = env.observation_spaces[good_agent_name[0]]
    action_space      = env.action_spaces[good_agent_name[0]]
    act_dim           = action_space.shape
    if comm:
        comm_space    = env.communication_spaces[good_agent_name[0]]
        if ablate_kwargs is None or ablate_kwargs['adv_train']:
            obs_dim       = (observation_space.shape[0]  + comm_space.shape[0],)
        else:
            num_agents = len(good_agent_name) + len(adv_agent_name)
            print("comm shape:{}, num_agent:{}".format(comm_space.shape[0], num_agents))
            obs_dim       = (observation_space.shape[0]  + 
                             comm_space.shape[0]*ablate_kwargs['k']//(num_agents-1),)
    else:
        obs_dim = env.observation_spaces[good_agent_name[0]].shape 
    if isinstance(action_space, Box):
        high = torch.from_numpy(action_space.high).to(Param.device).type(Param.dtype)
        low = torch.from_numpy(action_space.low).to(Param.device).type(Param.dtype)
    print("Obs dim:{}, Act dim:{}".format(obs_dim[0], act_dim))
    
    # Create actor-critic module
    ac = actor_critic(obs_dim[0], action_space, **ac_kwargs).to(Param.device)
    if trained_dir is not None:
        print("loaded pretrained model from", trained_dir)
        # state_dict, mean, std = torch.load(trained_dir, map_location=Param.device)
        state_dict, mean, std = torch.load(trained_dir, map_location=Param.device)
        ac.load_state_dict(state_dict)
        ac.moving_mean = mean
        ac.moving_std = std
        ### Set moving mean and average
        ac.MovingMeanStd.old_m = mean
        ac.MovingMeanStd.old_s = std
        ac.MovingMeanStd.n     = 40000
    
    # performance = test_return(env, ac, 50, 200, good_agent_name, 
    #                         dist_action, comm, recurrent, ablate_kwargs)
    # print("Initial 50 Episode Average Performance:{}".format(performance))
    # logger_file.write("Initial 50 Episode Average Performance:{}\n".format(performance))
    
    # Count variables
    var_counts = tuple(count_vars(module) for module in [ac.pi, ac.v])
    if not(verbose):
        print('--------------------\nNumber of parameters: \t pi: %d, \t v: %d\n------------------'%var_counts)
    
    # Set up experience buffer
    bufs = {}
    for agent in good_agent_name:
        bufs[agent] = PPOBuffer(obs_dim, act_dim, steps_per_epoch, gamma, lam)
    
    # Set up function for computing PPO policy loss
    def compute_loss_pi(data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']
        obs = ac.normalize(obs)
        act_info = dict(act_mean=torch.mean(act), act_std=torch.std(act))
        if beta:
            act = (act-low)/(high-low)
        pi, logp = ac.pi(obs, act)
            
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1+clip_ratio) | ratio.lt(1-clip_ratio)
        clipfrac = clipped.type(Param.dtype).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)
        return loss_pi, pi_info, act_info

    # Set up function for computing value loss
    def compute_loss_v(data):
        obs, ret = data['obs'], data['ret']
        obs = ac.normalize(obs)
        return ((ac.v(obs) - ret)**2).mean()
        
    # Set up optimizers for policy and value function
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)

    def update(data):
        pi_l_old, pi_info_old, act_info = compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = compute_loss_v(data).item()

        # Train policy with multiple steps of gradient descent
        for i in range(train_pi_iters):
            pi_optimizer.zero_grad()
            loss_pi, pi_info, act_info = compute_loss_pi(data)
            kl = pi_info['kl']
            if kl > 1.5 * target_kl and not verbose:
                print('Early stopping at step %d due to reaching max kl.'%i)
                break
            loss_pi.backward()
            pi_optimizer.step()
        stop_iter = i
        
        # Value function learning
        for i in range(train_v_iters):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            loss_v.backward()
            vf_optimizer.step()

        # Log changes from update
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
        act_mean, act_std = act_info['act_mean'], act_info['act_std']
        return dict(LossPi=pi_l_old, LossV=v_l_old,
                     KL=kl, Entropy=ent, ClipFrac=cf,
                     DeltaLossPi=(loss_pi.item() - pi_l_old),
                     DeltaLossV=(loss_v.item() - v_l_old),
                     StopIter=stop_iter,
                     Act_Mean=act_mean,
                     Act_Std=act_std)
    
    # Prepare for interaction with environment
    start_time = time.time()
    epoch_return, best_eval_return = [], -np.inf
    good_total_rewards = np.zeros(len(good_agent_name))
    best_eval_performance = -np.inf
    
    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        if comm:
            o, c = env.reset()
            o = concatenate(o,  c, ablate_kwargs)
        else:
            o = env.reset()
        ep_len, terminal = 0, False 
        avg_return, avg_len = [], []
        if adv_train:
            all_rewards = {a: [] for a in env.agents}
        
        ### Initialize hidden states for reucrrent network
        if recurrent:
            for agent in good_agent_name:
                ac.initialize()
            
        for t in range(steps_per_epoch):
            good_actions, values, log_probs = {},{},{}
            for agent in good_agent_name:
                a, v, logp = ac.step(torch.from_numpy(o[agent]).\
                                to(Param.device).type(Param.dtype), train=obs_normalize)
                values[agent] = v
                log_probs[agent] = logp
                if not dist_action:
                    a = a.cpu().numpy()
                    #a = np.clip(a, action_space.low, action_space.high)
                else:
                    a = a.item()
                good_actions[agent] = a
            if comm:
                next_o, c, reward, done, infos = env.step(good_actions)
                next_o = concatenate(next_o, c, ablate_kwargs)
            else:
                next_o, reward, done, infos = env.step(good_actions)
            if render:
                env.render()
            for i in range(len(good_agent_name)):
                agent = good_agent_name[i]
                if good_agent_name[i] in env.agents:
                    good_total_rewards[i] += reward[agent]
                if done[agent]:
                    terminal = True
                bufs[agent].store(o[agent], good_actions[agent], 
                                  reward[agent], values[agent], log_probs[agent])
            ep_len += 1
            # Update obs (critical!)
            o = next_o
            
            timeout = (ep_len == max_ep_len)
            terminal = (terminal or timeout)
            epoch_ended = (t== steps_per_epoch-1)

            if terminal or epoch_ended:
                if epoch_ended and not(terminal) and not(verbose):
                    print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
                # if trajectory didn't reach terminal state, bootstrap value target
                for agent in good_agent_name:
                    if timeout or epoch_ended:
                        _, v, _ = ac.step(torch.from_numpy(o[agent]).to(Param.device).type(Param.dtype), 
                                          train=obs_normalize)
                    else:
                        v = torch.tensor(0)
                    bufs[agent].finish_path(v.item())
                if terminal:
                    terminal = False
                    #ep_ret = np.mean(good_total_rewards)
                    i = int(re.match('pursuer_(\d+)', count_agent).group(1))
                    ep_ret = good_total_rewards[i]
                    good_total_rewards = np.zeros(len(good_total_rewards))
                    if adv_train:
                        for agent in env.agents:
                            all_rewards[agent].append(env.all_rewards[agent])
                   
                    avg_return.append(ep_ret)
                    avg_len.append(ep_len)
                if comm:
                    o, c = env.reset()
                    o = concatenate(o, c, ablate_kwargs)
                else:
                    o = env.reset()
                ### Initialize hidden states for reucrrent network
                if recurrent:
                    for agent in good_agent_name:
                        ac.initialize()
                ep_len = 0
        
        
        # Perform PPO update!
        agents_data = []
        for agent in good_agent_name:
            agents_data.append(bufs[agent].get())
        obs_buf  = torch.cat([data['obs'] for data in agents_data])
        act_buf  = torch.cat([data['act'] for data in agents_data])
        ret_buf  = torch.cat([data['ret'] for data in agents_data])
        adv_buf  = torch.cat([data['adv'] for data in agents_data])
        logp_buf = torch.cat([data['logp'] for data in agents_data])
        
        data = dict(obs=obs_buf, act=act_buf, ret=ret_buf,
                    adv=adv_buf, logp=logp_buf)
        param_dict = update(data)
        epoch_return.append(sum(avg_return)/len(avg_return))
            
        if not(verbose):
            print("----------------------Epoch {}----------------------------".format(epoch))
            logger_file.write("----------------------Epoch {}----------------------------\n".format(epoch))
            print("EpRet:{}".format(sum(avg_return)/len(avg_return)))
            logger_file.write("EpRet:{}\n".format(sum(avg_return)/len(avg_return)))
            if adv_train:
                print("Original Reward:", [(agent, np.mean(rew)) for agent, rew in all_rewards.items()])
            print("EpLen:{}".format(sum(avg_len)/len(avg_len)))
            logger_file.write("EpLen:{}\n".format(sum(avg_len)/len(avg_len)))
            print('V Values:{}'.format(v))
            logger_file.write('V Values:{}\n'.format(v))
            print('Total Interaction with Environment:{}'.format((epoch+1)*steps_per_epoch))
            logger_file.write('Total Interaction with Environment:{}\n'.format((epoch+1)*steps_per_epoch))
            print('LossPi:{}'.format(param_dict['LossPi']))
            logger_file.write('LossPi:{}\n'.format(param_dict['LossPi']))
            print('LossV:{}'.format(param_dict['LossV']))
            logger_file.write('LossV:{}\n'.format(param_dict['LossV']))
            print('DeltaLossPi:{}'.format(param_dict['DeltaLossPi']))
            logger_file.write('DeltaLossPi:{}\n'.format(param_dict['DeltaLossPi']))
            print('DeltaLossV:{}'.format(param_dict['DeltaLossV']))
            logger_file.write('DeltaLossV:{}\n'.format(param_dict['DeltaLossV']))
            print('Entropy:{}'.format(param_dict['Entropy']))
            logger_file.write('Entropy:{}\n'.format(param_dict['Entropy']))
            print('ClipFrac:{}'.format(param_dict['ClipFrac']))
            logger_file.write('ClipFrac:{}\n'.format(param_dict['ClipFrac']))
            print('KL:{}'.format(param_dict['KL']))
            logger_file.write('KL:{}\n'.format(param_dict['KL']))
            print('StopIter:{}'.format(param_dict['StopIter']))
            logger_file.write('StopIter:{}\n'.format(param_dict['StopIter']))
            print('Time:{}'.format(time.time()-start_time))
            logger_file.write('Time:{}\n'.format(time.time()-start_time))
            if adv_train:
                if env.default_policy is not None and env.default_policy.act_bias is not None:
                    print("Act Bias:{}".format(env.default_policy.act_bias))
                    print("Filter Count:{}".format(env.default_policy.filter_count))
                    logger_file.write("Filter Count:{}\n".format(env.default_policy.filter_count))
                    logger_file.write("Act Bias:{}\n".format(env.default_policy.act_bias))
                rew_file.write("Episode {}  EpRet:{}\n".format(epoch, -sum(avg_return)/len(avg_return)))
            else:
                rew_file.write("Episode {}  EpRet:{}\n".format(epoch, sum(avg_return)/len(avg_return)))
            rew_file.flush()
            logger_file.flush()
        
        ### Every few epochs, evaluate the current model's performance and save the model 
        ### if the eval performance is the best in the history.
        #if (log_freq is not None) and (epoch+1) % log_freq == 0:
            #performance,_ = test_return(env, ac, 50, 200, good_agent_name, dist_action, comm, recurrent, ablate_kwargs)
            #if performance > best_eval_performance:
                #best_eval_performance = performance
        if not no_save:
            ac.save(model_name='ppo_{}_{}'.format(name_env, exp_name))
                    
    logger_file.close()
    rew_file.close()
    
   

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
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epoch-smoothed', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--exp-name', type=str, default='ppo')
    parser.add_argument('--log-freq', type=int, default=50)
    parser.add_argument('--obs-normalize', action="store_true")
    parser.add_argument('--victim-no-beta', action="store_true")
    parser.add_argument('--beta', action="store_true")
    parser.add_argument('--trained-dir', type=str)
    parser.add_argument('--render', action="store_true")
    parser.add_argument('--truth', action="store_true")
    parser.add_argument('--smart', action="store_true")
    
    parser.add_argument('--count-agent', type=str, default='pursuer_0')
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
    
    parser.add_argument('--convert-adv', type=str, nargs='+')
    parser.add_argument('--good-policy', type=str)
    parser.add_argument('--victim', type=str, default="pursuer_0")
    
    parser.add_argument('--ablate', action='store_true')
    parser.add_argument('--ablate-k', type=int, default=1)
    parser.add_argument('--ablate-median', type=int, default=1)
    
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
    
    if args.detector_policy_dir is not None:
        confidence_kwargs = {'k':args.ablate_k,
                             'detector_policy_dir':args.detector_policy_dir} 
    else:
        confidence_kwargs = None
    
    if args.no_cuda:
        Param(torch.FloatTensor, torch.device("cpu"))
    else:
        Param(torch.cuda.FloatTensor, torch.device("cuda:{}".format(args.cuda)))
    
    ### Please make sure the following argument names are the same as the FoodCollector's init function
    if args.convert_adv:
        env = make_advcomm_env(adv_agents=args.convert_adv, good_policy_dir=args.good_policy, victim=args.victim,
                ac_kwargs=dict(hidden_sizes=[args.hid]*args.l, beta=not args.victim_no_beta, 
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
    
    logger_file = open(os.path.join(Param.data_dir, r"logger_{}.txt".format(args.exp_name)), "wt")
    logger_file.write("Number of Pursuers: {}  Number of Food:{}  Number of Poison:{}\n".\
                      format(args.n_pursuers, args.n_evaders, args.n_poison))
    logger_file.write("Number of Sensors: {}  Sensor Range:{}    Speed Features:{}\n".\
                      format(args.n_sensors, args.sensor_range, args.speed_features))
    logger_file.write("Food Speed: {}  Poison Speed:{}\n".\
                      format(args.evader_speed, args.poison_speed))
    logger_file.close()
        
    
    ppo(lambda:env, actor_critic=MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l, beta=args.beta, 
                       recurrent=args.recurrent, ep_len=args.max_cycle), 
        gamma=args.gamma, vf_lr=args.vf_lr, pi_lr=args.pi_lr,
        seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs, max_ep_len=args.max_cycle,
        name_env = args.env, epoch_smoothed=args.epoch_smoothed,
        no_save = args.no_save, verbose=args.verbose, log_freq = args.log_freq, 
        exp_name=args.exp_name, obs_normalize = args.obs_normalize, beta=args.beta,
        comm=args.comm, dist_action=args.dist_action, trained_dir=args.trained_dir, 
        render=args.render, recurrent=args.recurrent, adv_train=True if args.convert_adv else False,
        ablate_kwargs=ablate_kwargs, count_agent=args.count_agent)
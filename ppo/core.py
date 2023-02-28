from param import Param
import numpy as np
import scipy.signal
from gym.spaces import Box, Discrete
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.beta import Beta
from torch.distributions.categorical import Categorical
from torch.nn import GRU, Linear
import copy


# Return all combinations of size k of numbers
def makeCombi(n, k):
    def makeCombiUtil(n, left, k, tmp, idx):
        if (k == 0):
            idx.append(random.sample(copy.deepcopy(tmp), len(tmp)))
            return
        for i in range(left, n + 1):
            tmp.append(i-1)
            makeCombiUtil(n, i + 1, k - 1, tmp, idx)
            tmp.pop()
    tmp, idx = [], []
    makeCombiUtil(n, 1, k, tmp, idx)
    return idx

def from_numpy(n_array, dtype=None):
    if dtype is None:
        return torch.from_numpy(n_array).to(Param.device).type(Param.dtype)
    else:
        return torch.from_numpy(n_array).to(Param.device).type(dtype)

    
def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input: 
        vector x, 
        [x0, 
         x1, 
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

### Clip the Action
def clip(action, low, high):
    return np.clip(action, low, high)

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

class Actor(nn.Module):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and 
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a
    
class MLPCategoricalActor(Actor):
    
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, recurrent=False, ep_len=1000):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class MLPGaussianActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, 
                 recurrent=False, ep_len=1000):
        super().__init__()
        self.obs_dim   = obs_dim
        self.recurrent = recurrent
        self.ep_len    = ep_len
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        if not recurrent:
            self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)
        else:
            self.num_layers = len(hidden_sizes)
            self.hidden_size= hidden_sizes[0]
            self.mu_gru  = GRU(obs_dim, hidden_sizes[0], len(hidden_sizes), batch_first=True)
            self.mu_net  = Linear(hidden_sizes[0], act_dim)
    
    ### Initialize latent state, called only when recurrent is True
    def initialize(self):
        self.h = torch.zeros(self.num_layers, 1, self.hidden_size).to(Param.device).type(Param.dtype)
    
    def _distribution(self, obs, mu_h=None):
        std = torch.exp(self.log_std)
        if not self.recurrent:
            mu = self.mu_net(obs)
        else:
            if len(obs.shape) == 1:
                obs            = obs.reshape(1,1,-1)
                mu, self.h = self.mu_gru(obs, self.h)
                mu       = mu.reshape(-1)
                mu       = self.mu_net(mu)
            else:
                batch_size = obs.shape[0]//self.ep_len
                obs = obs.reshape(batch_size, self.ep_len, -1)
                h   = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(Param.device).type(Param.dtype)
                mu, _ = self.mu_gru(obs, h)
                mu   = mu.reshape(-1, self.hidden_size)
                mu   = self.mu_net(mu)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)    # Last axis sum needed for Torch Normal distribution
    
class MLPBetaActor(Actor):
    
    ### Beta distribution, dealing with the case where action is bounded in the 
    ### box (-epsilon, epsilon)
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, high, recurrent=False, ep_len=1000):
        super().__init__()
        self.high = high
        self.ep_len = ep_len
        self.recurrent = recurrent
        self.obs_dim   = obs_dim
        if not recurrent:
            self.alpha = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)
            self.beta  = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)
        else:
            self.num_layers = len(hidden_sizes)
            self.h_size = (self.num_layers, 1, hidden_sizes[0])
            self.hidden_size= hidden_sizes[0]
            self.alpha_gru  = GRU(obs_dim, hidden_sizes[0], len(hidden_sizes), batch_first=True)
            self.beta_gru   = GRU(obs_dim, hidden_sizes[0], len(hidden_sizes), batch_first=True)
            self.alpha      = Linear(hidden_sizes[0], act_dim)
            self.beta       = Linear(hidden_sizes[0], act_dim)
    
    ### Initialize latent state, called only when recurrent is True
    def initialize(self):
        self.alpha_h = torch.zeros(self.num_layers, 1, self.hidden_size).to(Param.device).type(Param.dtype)
        self.beta_h  = torch.zeros(self.num_layers, 1, self.hidden_size).to(Param.device).type(Param.dtype)
        
    ### Input shape: (obs_dim,) or (batch_size, obs_dim)
    def _distribution(self, obs):
        if not self.recurrent:
            alpha = self.alpha(obs)
            beta  = self.beta(obs)
        else:   
            if len(obs.shape) == 1:
                obs            = obs.reshape(1,1,-1)
                alpha, self.alpha_h = self.alpha_gru(obs, self.alpha_h)
                beta,  self.beta_h  = self.beta_gru(obs,  self.beta_h)
                alpha, beta    = alpha.reshape(-1), beta.reshape(-1)
            else:
                batch_size = obs.shape[0]//self.ep_len
                obs = obs.reshape(batch_size, self.ep_len, -1)
                alpha_h = torch.zeros(self.num_layers, batch_size, self.hidden_size).\
                        to(Param.device).type(Param.dtype)
                beta_h  = torch.zeros(self.num_layers, batch_size, self.hidden_size).\
                        to(Param.device).type(Param.dtype)
                alpha, _ = self.alpha_gru(obs, alpha_h)
                beta,  _  = self.beta_gru(obs, beta_h)
                alpha, beta    = alpha.reshape(-1, self.hidden_size), beta.reshape(-1, self.hidden_size)
            alpha = self.alpha(alpha)
            beta  = self.beta(beta)
        alpha = torch.log(1+torch.exp(alpha))+1
        beta  = torch.log(1+torch.exp(beta))+1
        return Beta(alpha, beta)
    
    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)   
    
class MLPCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation, recurrent=False, ep_len=1000):
        super().__init__()
        self.ep_len = ep_len
        self.recurrent = recurrent
        if recurrent:
            self.num_layers = len(hidden_sizes)
            self.hidden_size= hidden_sizes[0]
            self.v_gru  = GRU(obs_dim, hidden_sizes[0], len(hidden_sizes), batch_first=True)
            self.v      = Linear(hidden_sizes[0], 1)
        else:
            self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)
    
    ### Initialize latent state, called only when recurrent is True
    def initialize(self):
        self.h = torch.zeros(self.num_layers, 1, self.hidden_size).to(Param.device).type(Param.dtype)
    
    def forward(self, obs, v_h=None):
        if not self.recurrent:
            return torch.squeeze(self.v_net(obs), -1) # Critical to ensure v has right shape
        else:
            if len(obs.shape) == 1:
                obs            = obs.reshape(1,1,-1)
                v, self.h = self.v_gru(obs, self.h)
                v       = v.reshape(-1)
            else:
                batch_size = obs.shape[0]//self.ep_len
                obs = obs.reshape(batch_size, self.ep_len, -1)
                h   = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(Param.device).type(Param.dtype)
                v, _ = self.v_gru(obs, h)
                v   = v.reshape(-1, self.hidden_size)
            return torch.squeeze(self.v(v), -1)
    
    
class MLPActorCritic(nn.Module):

    def __init__(self, obs_dim, action_space, hidden_sizes=(64,64), activation=nn.Tanh, beta=False, recurrent=False, ep_len=1000):
        super().__init__()
        
        self.beta = beta ### Whether to use beta distribution to deal with clipped action space
        self.recurrent = recurrent
        # policy builder depends on action space
        if isinstance(action_space, Box) and not beta:
            self.pi = MLPGaussianActor(obs_dim, action_space.shape[0], hidden_sizes, 
                                       activation, recurrent, ep_len)
        elif isinstance(action_space, Discrete):
            self.beta = False
            self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, 
                                          activation, recurrent, ep_len)
        else:
            self.high = torch.from_numpy(action_space.high).type(Param.dtype).to(Param.device)
            self.low = torch.from_numpy(action_space.low).type(Param.dtype).to(Param.device)
            self.pi = MLPBetaActor(obs_dim, action_space.shape[0], 
                                   hidden_sizes, activation, self.high, 
                                   recurrent, ep_len)
        # build value function
        self.v  = MLPCritic(obs_dim, hidden_sizes, activation, recurrent, ep_len)
        
        self.MovingMeanStd = MovingMeanStd((obs_dim,))
        self.moving_mean = torch.zeros(obs_dim).to(Param.device).type(Param.dtype)
        self.moving_std  = torch.ones(obs_dim).to(Param.device).type(Param.dtype)
    
    def initialize(self):
        self.v.initialize()
        self.pi.initialize()
        
    def step(self, obs, train=False):
        with torch.no_grad():
            if train:
                self.MovingMeanStd.push(obs)
                self.moving_mean = self.MovingMeanStd.mean()
                self.moving_std  = self.MovingMeanStd.std()
            obs = (obs - self.moving_mean)/(self.moving_std+1e-6)
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            if self.beta:
                a = a*(self.high-self.low)+self.low ### Clip to the correct range
            v = self.v(obs)
        return a, v, logp_a

    def act(self, obs):
        return self.step(obs)[0]
        
    def save(self, log_dir='./learned_models/', model_name='ppo_policy'):
        torch.save([self.state_dict(), self.moving_mean, self.moving_std], os.path.join(log_dir,model_name))
    
    ### Return Normalized Observation
    def normalize(self, obs):
        return (obs - self.moving_mean)/(self.moving_std+1e-6)
    
class RandomAttacker(nn.Module):

    def __init__(self, action_space):
        super().__init__()
        self.act_dim  = action_space.shape
        self.act_high = from_numpy(action_space.high)
        self.act_low  = from_numpy(action_space.low)
        
    def act(self, obs=None):
        act = torch.randint(high=2, size=self.act_dim).to(Param.device)
        return torch.where(act==0, self.act_low, self.act_high)
    
def shuffle(comm, k):
    choice = np.random.choice(comm.shape[0], k, replace=False)
    return comm[choice]


def concatenate(obs, comm, ablate_kwargs=None, idx_list=None):
    for agent in obs:
        if ablate_kwargs is None or agent in ablate_kwargs["adv_agents"]:
            obs[agent] = np.concatenate([obs[agent], comm[agent]])
        else:
            choice = np.random.choice(idx_list.shape[0], ablate_kwargs['median'], 
                                     replace=False)
            idxs = [idx_list[i] for i in choice]
            o  = np.stack([obs[agent]]*ablate_kwargs['median'])
            comm_agent  = comm[agent].reshape(len(obs)-1, -1)
            comm_agent  = np.concatenate([comm_agent[idx].reshape(1,-1) for idx in idxs])
            obs[agent]  = np.concatenate([o, comm_agent], axis=-1)
    return obs
## Train adversary
## default: benign agents -> shuffle

def test_return(env, ac, num_t, max_steps, good_agent_name, dist_action, comm, 
                recurrent=False, ablate_kwargs=None, random_ac=False):
    action_space = env.action_spaces['pursuer_1']
    if isinstance(action_space, Box):
        low, high = action_space.low, action_space.high
    else:
        low, high = None, None
        
    agent_rewards = [[]]*len(good_agent_name)
    if ablate_kwargs is not None:
        idx_list = np.array(makeCombi(ablate_kwargs['n']-1, ablate_kwargs['k']))
    else:
        idx_list  = None
    for eps in range(num_t):
        episode_rewards = np.zeros(len(good_agent_name))
        if recurrent:
            for agent in good_agent_name:
                ac.initialize()
        if comm:
            o, c = env.reset()
            o = concatenate(o,  c, ablate_kwargs, idx_list)
        else:
            o = env.reset()
        done = False
        for t in range (max_steps):
            good_actions = {}
            for agent in good_agent_name:
                if not random_ac:
                    a = ac.act(torch.from_numpy(o[agent]).to(Param.device).type(Param.dtype))
                    if dist_action:
                        a = torch.mode(a, axis=-1)[0]
                    elif len(a.shape)>1 and not dist_action:
                        a = torch.median(a, axis=0)[0]
                    if not dist_action:
                        a = a.cpu().numpy()
                        a = clip(a, low, high)
                    else:
                        a = a.item()
                else:
                    a = action_space.sample()
                good_actions[agent] = a
            if comm:
                next_o, c, reward, dones,_  = env.step(good_actions)
                next_o = concatenate(next_o, c, ablate_kwargs, idx_list)
            else:
                next_o, reward, dones,_  = env.step(good_actions)
            for i in range(len(good_agent_name)):
                agent = good_agent_name[i]
                if dones[agent]:
                    done = True
                episode_rewards[i] += reward[agent]
            o = next_o
            if done:
                for i in range(len(good_agent_name)):
                    agent_rewards[i].append(episode_rewards[i])
                break
    reward = np.array(agent_rewards[0])
    reward = np.mean(reward.reshape(3, reward.shape[0]//3), axis=-1)
    return np.mean(reward), np.std(reward)




### Calculating moving meana and standard deviation
class MovingMeanStd:

    def __init__(self, shape):
        self.n = 0
        self.old_m = 0
        self.new_m = 0
        self.old_s = 0
        self.new_s = 0
        self.shape = shape

    def clear(self):
        self.n = 0

    def push(self, x):
        self.n += 1

        if self.n == 1:
            self.old_m = self.new_m = x
            self.old_s = 0
        else:
            self.new_m = self.old_m + (x - self.old_m) / self.n
            self.new_s = self.old_s + (x - self.old_m) * (x - self.new_m)

            self.old_m = self.new_m
            self.old_s = self.new_s

    def mean(self):
        if self.n > 0:
            return self.new_m
        else:
            return torch.zeros(self.shape).to(Param.device).type(Param.dtype)

    def variance(self):
        if self.n > 1:
            return self.new_s / (self.n - 1) 
        else:
            return torch.ones(self.shape).to(Param.device).type(Param.dtype)

    def std(self):
        return torch.sqrt(self.variance())

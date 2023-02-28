# Certifiably Robust Policy Learning against Adversarial Multi-Agent Communication



## 1. Introduction

This is a python implementation for our AME method and baselines. The environment implementation is modified from the [PettingZoo](https://pettingzoo.farama.org/)[1] MARL codebase (in folder *PettingZoo*). The PPO trainer is modified from the [SpinningUp](https://spinningup.openai.com/en/latest/) codebase by OpenAI (in folder *ppo*).

We suggest creating a new virtual environment (conda is suggested with python version 3.7), and run the following command to install the required packages 

```
pip install -r requirements.txt 
cd PettingZoo
pip install -e .
cd ../
cd ppo
```

We require PyTorch to train agents, so please also install PyTorch based on your hardware. For example, if you are using Linux and GPU with CUDA 10.2, please run

```
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
```


## 2. Training and Evaluating AME Agents

### For Discrete Action Space 

- Train a message-ablation policy of 9 agents in a benign setting with communication:

```
python ppo.py --epochs 800 --exp-name NAME_OF_THE_MODEL --obs-normalize --comm --smart --cuda CUDA --n-pursuers 9 --ablate --ablate-k 2 --dist-action
# --ablate-k is the ablation size hyperparameter (i.e., $k$ in the paper)
# --n-pursuers is the number of agents in this game (i.e., $N$ in the paper)
# To train a vanilla agent without AME certification, set --ablate-k as N-1=8.
# If gpu is not available, please specify --no-cuda
# To train a policy without communication signal, remove --comm and --smart
```

Note that increasing ablation size k will increase the clean performance, while dropping the robust performance.

The log and reward files can be found under ./data/. Trained model is saved in ./learned_models/ as ppo_FoodCollector_${NAME_OF_THE_MODEL}


- Evaluate the message-ensemble policy under no attack (clean environment):
```
python test.py --epochs 200 --trained-dir ./learned_models/ppo_FoodCollector_${NAME_OF_THE_MODEL} --obs-normalize --comm --smart --cuda CUDA --n-pursuers 9 --ablate --ablate-k 2 --ablate-median 28 --dist-action
# --ablate-median is the sample size D, which is not larger than \binom{N-1}{k}, and a larger D is prefered for robustness.  
# For a vanilla agent in this scenario, set --ablate-k 8 --ablate-median 1
```

- Evaluate the message-ensemble policy under random attack (heuristic attack):
```
python test.py --epochs 200 --trained-dir ./learned_models/ppo_FoodCollector_${NAME_OF_THE_MODEL} --obs-normalize --comm --smart --cuda CUDA --n-pursuers 9 --ablate --ablate-k 2 --ablate-median 28 --victim-dist-action --test-random-attacker
```

- Evaluate the message-ensemble policy of 9 agents under a learned attacker (strong adaptive attack):

```
python ppo.py --epochs 800 --exp-name NAME_OF_ATTACKER_MODEL --obs-normalize --comm --smart --cuda CUDA --victim pursuer_0 --convert-adv pursuer_1 --pursuer_2 --good-policy ./learned_models/ppo_FoodCollector_${NAME_OF_VICTIM_MODEL}$ --n-pursuers 9 --ablate --ablate-k 2 --ablate-median 28 --victim-dist-action
# train an attacker against the fixed victim policy
# The directory after --good-policy is where the pre-trained victim model gets stored
```



### For Continuous Action Space


- Train the message-ablation policy of 9 agents in a benign setting with communication:

```
python ppo.py --epochs 800 --exp-name NAME_OF_THE_MODEL --obs-normalize --comm --smart --cuda CUDA --n-pursuers 9 --ablate --ablate-k 2 --beta
```

- Evaluate the message-ensemble policy under no attack (clean environment):

``` 
python test.py --epochs 200 --trained-dir ./learned_models/ppo_FoodCollector_${NAME_OF_THE_MODEL} --obs-normalize --comm --smart --cuda CUDA --n-pursuers 9 --ablate --ablate-k 8 --beta
```

- Evaluate the message-ensemble policy under random attack (heuristic attack):
```
python test.py --epochs 200 --trained-dir ./learned_models/ppo_FoodCollector_${NAME_OF_THE_MODEL} --obs-normalize --comm --smart --cuda CUDA --n-pursuers 9 --ablate --ablate-k 2 --ablate-median 28 --test-random-attacker
```

- Evaluate the message-ensemble policy of 9 agents under a learned attacker (strong adaptive attack):

```
python ppo.py --epochs 800 --exp-name NAME_OF_ATTACKER_MODEL --obs-normalize --comm --smart --cuda CUDA --victim pursuer_0 --convert-adv pursuer_1 --pursuer_2 --good-policy ./learned_models/ppo_FoodCollector_${NAME_OF_VICTIM_MODEL}$ --n-pursuers 9 --ablate --ablate-k 2 --ablate-median 28 --victim-dist-action
```




References:

[1] J. K Terry, Benjamin Black, Nathaniel Grammel, Mario Jayakumar, Ananth Hari, Ryan Sulivan, Luis Santos, Rodrigo Perez, Caroline Horsch, Clemens Dieffendahl, Niall L Williams, Yashas Lokesh, Ryan Sullivan, and Praveen Ravi. Pettingzoo: Gym for multi-agent reinforcement learning. arXiv preprint arXiv:2009.14471, 2020

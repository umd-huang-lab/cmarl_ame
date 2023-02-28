# Certifiably Robust Policy Learning against Adversarial Multi-Agent Communication



## 1. Introduction

This is a python implementation for our AME method and baselines. The environment implementation is modified from the PettingZoo[1] MARL codebase (in folder *PettingZoo*). The PPO trainer is modified from the SpinningUp codebase by OpenAI (in folder *ppo*).

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


## 2. Train vanilla/robust agent

- Train a policy of 9 agents in a benign setting with communication (discrete action):

  The log and reward files can be found under ./data/
```
python ppo.py --epochs 800 --exp-name NAME_OF_THE_MODEL --obs-normalize --comm --smart --cuda CUDA --n-pursuers 9 --ablate --ablate-k 8 --dist-action
```
*If gpu is not available, please specify --no-cuda*

*To train a policy without communication signal, remove --comm and --smart*

Here randomized ablation size k is set to be 8, so that every agent is taking every communication message from each other agent. To train a randomized ablation policy with different ablation size k, change --ablate-k. Increase ablation size k will increase the clean performance, while dropping the robust performance.

Trained model is saved in ./learned_models/ as ppo_FoodCollector_${NAME_OF_THE_MODEL}



- Train a policy of 9 agents in a benign setting with communication (continuous action):

```
python ppo.py --epochs 800 --exp-name NAME_OF_THE_MODEL --obs-normalize --comm --smart --cuda CUDA --n-pursuers 9 --ablate --ablate-k 8 --beta
```



- Evaluate the trained policy with 200 episodes (discrete action):

```
python test.py --epochs 200 --trained-dir ./learned_models/ppo_FoodCollector_${NAME_OF_THE_MODEL} --obs-normalize --comm --smart --cuda CUDA --n-pursuers 9 --ablate --ablate-k 8 --dist-action
```



- Evaluate the trained policy with 200 episodes (continuous action)

``` 
python test.py --epochs 200 --trained-dir ./learned_models/ppo_FoodCollector_${NAME_OF_THE_MODEL} --obs-normalize --comm --smart --cuda CUDA --n-pursuers 9 --ablate --ablate-k 8 --beta
```



## 3. Train Attacker

- Train an adversary agent against a fixed victim (vanilla training, i.e. k=8)

```
python ppo.py --epochs 800 --exp-name NAME_OF_ATTACKER_MODEL --obs-normalize --comm --smart --cuda CUDA --victim pursuer_0 --convert-adv pursuer_1 --pursuer_2 --good-policy ./learned_models/ppo_FoodCollector_${NAME_OF_VICTIM_MODEL}$ --n-pursuers 9 --ablate --ablate-k 8 --victim-dist-action
```

*To train an attacker with continuous action space, remove --dist-action*

*The directory after --good-policy is where the pre-trained victim model gets stored*



- Train an adversary agent against randomized ablation defense

```
python ppo.py --epochs 800 --exp-name NAME_OF_ATTACKER_MODEL --obs-normalize --comm --smart --cuda CUDA --victim pursuer_0 --convert-adv pursuer_1 --pursuer_2 --good-policy ./learned_models/ppo_FoodCollector_${NAME_OF_VICTIM_MODEL}$ --n-pursuers 9 --ablate --ablate-k K --ablate-median D --victim-dist-action
```

*To train an attacker with continuous action space, remove --dist-action*

Here, ablation size K and ablation sample size D are two hyperparameters for randomized ablation defensive algorithm. Note that $D\leq\binom{N-1}{K}$, where $N$ is the total number of agents. See the paper for a detailed discussion on how to choose K and D. 



- Evaluate the trained attacker with 200 episodes:

```
python test.py --epochs 200 --trained-dir ./learned_models/ppo_FoodCollector_${NAME_OF_ATTACKER_MODEL} --obs-normalize --comm --smart --cuda CUDA --n-pursuers 9 --victim pursuer_0 --convert-adv pursuer_1 --pursuer_2 --good-policy ./learned_models/ppo_FoodCollector_${NAME_OF_VICTIM_MODEL}$ --ablate --ablate-k K --ablate-median D --dist-action
```

*To train an attacker with continuous action space, remove --dist-action*

*To evaluate the trained attacker against vanilla model, set K=8 and D=1*



## 4. Adversarial Training

We provide the shell script for adversarial training with 9 agents and 2 adversaries in ./scripts/atla_n9_adv2.sh (continuous action space) and ./scripts/atla_n9_adv2_dist.sh (discrete action space).

To execute the one for continuous action space, type

``` 
bash ./scripts/atla_n9_m2.sh CUDA SEED &
```

*If gpu is not available, please specify --no-cuda in the shell script*

To execute the one for continuous action space, type

``` 
bash ./scripts/atla_n9_m2_dist.sh CUDA SEED &
```



Here we alternate the training of victim and adversary after 200 epochs, which counts as 1 iteration. After i iterations, the trained robust model will be saved as ./learned_models/ppo_FoodCollector_atla_agent9_victim_adv2_iter\$i_dist_\$SEED

To evaluate the trained robust model, follow the instruction above to train the attacker and evaluate its clean/robust performance.





References:

[1] J. K Terry, Benjamin Black, Nathaniel Grammel, Mario Jayakumar, Ananth Hari, Ryan Sulivan, Luis Santos, Rodrigo Perez, Caroline Horsch, Clemens Dieffendahl, Niall L Williams, Yashas Lokesh, Ryan Sullivan, and Praveen Ravi. Pettingzoo: Gym for multi-agent reinforcement learning. arXiv preprint arXiv:2009.14471, 2020
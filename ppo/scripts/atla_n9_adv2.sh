CUDA=$1
SEED=$2

### Pretrain a Victim model
python train_atla.py --steps 4000 --epochs 500 --exp-name atla_agent9_victim_adv2_iter0_${SEED} --obs-normalize --cuda ${CUDA} --beta --comm --smart --evader-speed 0 --poison-speed 0 --n-sensors 6 --n-pursuers 9 --n-evaders 1 --n-poison 1 --max-cycle 200 & process_id1=$! 
wait $process_id1

for i in {1..10}
do
    echo "Iteration: $i"
    let j=i-1
    if [ $i == 1 ]
    
    ### Train Attacker Against Victim
    then
    python train_atla.py --steps 4000 --epochs 200 --exp-name atla_agent9_adv2_iter${i}_${SEED} --obs-normalize --cuda ${CUDA} --comm --smart --evader-speed 0 --poison-speed 0 --n-sensors 6 --n-pursuers 9 --n-evaders 1 --n-poison 1 --max-cycle 200 --convert-adv pursuer_1 pursuer_2 --victim pursuer_0 --good-policy ./learned_models/ppo_FoodCollector_atla_agent9_victim_adv2_iter${j}_${SEED}  & process_id1=$! 
wait $process_id1
    else
    python train_atla.py --steps 4000 --epochs 200 --exp-name atla_agent9_adv2_iter${i}_${SEED} --obs-normalize --cuda ${CUDA} --comm --smart --evader-speed 0 --poison-speed 0 --n-sensors 6 --n-pursuers 9 --n-evaders 1 --n-poison 1 --max-cycle 200 --convert-adv pursuer_1 pursuer_2 --victim pursuer_0 --adv-policy ./learned_models/ppo_FoodCollector_atla_agent9_adv2_iter${j}_${SEED} --good-policy ./learned_models/ppo_FoodCollector_atla_agent9_victim_adv2_iter${j}_${SEED} & process_id1=$! 
wait $process_id1

    ### Train Victim Against Attacker
    fi
    python train_atla.py --steps 4000 --epochs 200 --exp-name atla_agent9_victim_adv2_iter${i}_${SEED} --obs-normalize --cuda ${CUDA} --beta --comm --smart --evader-speed 0 --poison-speed 0 --n-sensors 6 --n-pursuers 9 --n-evaders 1 --n-poison 1 --max-cycle 200 --convert-adv pursuer_1 pursuer_2 --victim pursuer_0 --adv-policy ./learned_models/ppo_FoodCollector_atla_agent9_adv2_iter${i}_${SEED} --good-policy ./learned_models/ppo_FoodCollector_atla_agent9_victim_adv2_iter${j}_${SEED} --train-victim & process_id1=$!
wait $process_id1    

done


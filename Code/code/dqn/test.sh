#!/bin/bash

if [ -z "$1" ]
  then echo "Please provide the port for the game, e.g.  ./run_cpu.sh 5050 "; exit 0
fi

# dataset='CoNLLTest_R'
# dataset='AKWS1News'
# dataset='Ontonotes_EN'
# dataset='Ontonotes_EN_Sub_R'
# dataset='IEER'

#classifierModel='AdaBoostM1'
#classifierModel='Bagging'
#classifierModel='DT_J48'
#classifierModel='LogisticRegression'
#classifierModel='MultilayerPerceptron'
#classifierModel='NaiveBayes'
#classifierModel='RandomForest'
#classifierModel='SVM_LibSVM'

#port=6003

dataset='$1'
classifierModel='$2'
port='$3'

echo 'dataset        =$dataset'
echo 'classifierModel=$classifierModel1'
echo 'port           =$port'

mode='NER_SIA'
# mode='NER_SPLON'
if [ "$mode" == "NER_SIA" ]
then
    state_dim=21;
    n_queries=2;
else
    state_dim=31;
    n_queries=4;
fi;

steps=2000000
eval_freq=10000

if [ "$dataset" == "CoNLLTest_R" ]; then
    eval_episodes=4863
elif [ "$dataset" == "AKWS1News" ]; then
    eval_episodes=9686
elif [ "$dataset" == "Ontonotes_EN" ]; then
   eval_freq=50000
   eval_episodes=58641
elif [ "$dataset" == "Ontonotes_EN_Sub_R" ]; then
   eval_episodes=11054
   # eval_episodes=6661    # filter
   # eval_episodes=8305    # t/e reverse
   # eval_episodes=2914    # query by ctx word
   # eval_episodes=2829    # query by ctx word, t/e reverse
elif [ "$dataset" == "IEER" ]; then
   eval_episodes=2137
else
    eval_episodes=1000
fi;

# exp_folder=$2
exp_folder="/mnt/hgfs/VMWareShare/"$dataset"/"$classifierModel"/"

FRAMEWORK="alewrap"
game_path=$PWD"/roms/"
env_params="useRGB=true"
agent="NeuralQLearner"
n_replay=1

netfile="\"network_mass\""
# netfile="\"logs/tmp2/agent_5051.t7\""
# netfile="\"logs/tmp2_2/agent_5050.t7\""
minibatch_size=100
update_freq=1
actrep=1 #TODO: check this
discount=${3:-0.8}
seed=1

#learn_start=${4:-10000}
learn_start=${4:-50000}

pool_frms_type="\"max\""
pool_frms_size=2
initial_priority="false"
replay_memory=500000
# replay_memory=5000
# replay_memory=1000000
eps_end=0.1
eps_endt=$replay_memory
lr=${5:-0.000025}
lr_end=0.000025
lr_endt=$replay_memory
wc=0.0
agent_type="DQN3_0_1"
agent_name="agent_"$port
ncols=1
target_q=${6:-5000}

agent_params="n_queries="$n_queries",wc="$wc",lr="$lr",lr_end="$lr_end",lr_endt="$lr_endt",ep=1,ep_end="$eps_end",ep_endt="$eps_endt",discount="$discount",hist_len=1,learn_start="$learn_start",replay_memory="$replay_memory",update_freq="$update_freq",n_replay="$n_replay",network="$netfile",state_dim="$state_dim",minibatch_size="$minibatch_size",rescale_r=1,ncols="$ncols",bufferSize=512,valid_size=500,target_q="$target_q",clip_delta=10,min_reward=-10,max_reward=10"

prog_freq=10000
save_freq=10000
gpu=-1
random_starts=0
pool_frms="type="$pool_frms_type",size="$pool_frms_size
num_threads=4

args="-mode $mode -exp_folder $exp_folder -zmq_port $port -framework $FRAMEWORK -game_path $game_path -name $agent_name -env_params $env_params -agent $agent -agent_params $agent_params -steps $steps -eval_freq $eval_freq -eval_episodes $eval_episodes -prog_freq $prog_freq -save_freq $save_freq -actrep $actrep -gpu $gpu -random_starts $random_starts -pool_frms $pool_frms -seed $seed -threads $num_threads"
echo $args

cd dqn
mkdir -p $exp_folder;
mkdir -p $exp_folder'/tmp';
OMP_NUM_THREADS=4 th train_agent.lua $args      #lutm: call train_agent.lua

#!/bin/bash

if [ -z "$1" ]
  then echo "Please provide the port for the game, e.g.  ./run_cpu.sh 5050 "; exit 0
fi


dataset="$1"
classifierModel="$2"
port="$3"
expName="$4"
state_dim="$5"
eval_episodes="$6"
mode="$7"
n_hid1="$8"
n_hid2="$9"
discount="${10}"

echo "dataset        =$dataset"
echo "classifierModel=$classifierModel"
echo "port           =$port"
echo "expName        =$expName"
echo "state_dim      =$state_dim"
echo "n_hid1         =$n_hid1"
echo "n_hid2         =$n_hid2"
echo "discount       =$discount"

#steps=2000000
steps=1050000
eval_freq=10000
n_queries=3

# exp_folder=$2
exp_folder="/mnt/hgfs/VMWareShare/"$expName"/"$dataset"/"$classifierModel"/"

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
#discount=${3:-0.8}
discount=$discount
seed=1

#learn_start=${4:-10000}
learn_start=50000

pool_frms_type="\"max\""
pool_frms_size=2
initial_priority="false"
replay_memory=500000
# replay_memory=5000
# replay_memory=1000000
eps_end=0.1
eps_endt=$replay_memory
lr=0.000025
lr_end=0.000025
lr_endt=$replay_memory
wc=0.0
agent_type="DQN3_0_1"
agent_name="agent_"$port
ncols=1
target_q=5000

agent_params="n_queries="$n_queries",wc="$wc",lr="$lr",lr_end="$lr_end",lr_endt="$lr_endt",ep=1,ep_end="$eps_end",ep_endt="$eps_endt",discount="$discount",hist_len=1,learn_start="$learn_start",replay_memory="$replay_memory",update_freq="$update_freq",n_replay="$n_replay",network="$netfile",state_dim="$state_dim",minibatch_size="$minibatch_size",rescale_r=1,ncols="$ncols",bufferSize=512,valid_size=500,target_q="$target_q",clip_delta=10,min_reward=-10,max_reward=10,n_hid1="$n_hid1",n_hid2="$n_hid2

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
#mkdir -p $exp_folder'/tmp';
OMP_NUM_THREADS=4 th train_agent.lua $args      #lutm: call train_agent.lua

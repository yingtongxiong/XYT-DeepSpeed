#!/bin/bash
set -x

export PYTHONPATH=/mnt/petrelfs/xiongyingtong/XYT-DeepSpeed:$PYTHONPATH

# export NCCL_DEBUG=info
export NCCL_SOCKET_IFNAME="bond0"
export NCCL_IB_HCA="mlx5_2,mlx5_3,mlx5_4,mlx5_5"

export CUDA_DEVICE_MAX_CONNECTIONS=1

# launch preparation
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

head_node=${nodes_array[0]}

head_node_ip=$(cat /etc/hosts | grep -w "$head_node" | awk '{print $1}')
echo $head_node

## env config
GPUS_PER_NODE=8
MASTER_ADDR=$head_node_ip
MASTER_PORT=7880
NNODES=$SLURM_NNODES

MODEL_ARGS=" \
    --num-layers 2 \
    --hidden-size 4096 \
    --num-attention-heads 32 \
    --vocab-size 50304 \
    --mlp-ratio 2.5 \
    --dtype torch.bfloat16 \
    --parallel-output False \
"

DATA_ARGS=" \
    --seq-len 4096 \
    --packed-length 4096 \
    --micro-bsz 1 \
    --micro-num 1 \
    --rampup-batch-size 1 \
    --fixed_random_dataset_seqlen True \
    --pack_sample_into_one False \
    --num-worker 4 \
"

DEEPSPEED_ARGS="\
    --deepspeed \
    --deepspeed_config deepspeed_config.json \
"


torchrun --nproc_per_node 8 --nnodes $NNODES --node_rank $SLURM_PROCID --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
pretrain.py $MODEL_ARGS $DATA_ARGS $DEEPSPEED_ARGS

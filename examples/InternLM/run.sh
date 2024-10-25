#!/bin/bash
set -x

# 指定从哪里导入deepspeed，通过无需安装的方式，可以直接使用自己修改的deepspeed
export PYTHONPATH=/mnt/petrelfs/xiongyingtong/XYT-DeepSpeed:$PYTHONPATH

# NCCL相关环境变量
# export NCCL_DEBUG=info
export NCCL_SOCKET_IFNAME="bond0"
export NCCL_IB_HCA="mlx5_2,mlx5_3,mlx5_4,mlx5_5"

# CUDA相关环境变量
export CUDA_DEVICE_MAX_CONNECTIONS=1

# 获取slurm相关变量
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

# data相关config
micro_batch_size=1
gradient_accumulate_size=1

seq_len=4096


# 模型相关的config
MODEL_ARGS=" \
    --num-layers 32 \
    --hidden-size 4096 \
    --num-attention-heads 32 \
    --vocab-size 103168 \
    --mlp-ratio 8/3 \
    --dtype torch.bfloat16 \
    --parallel-output False \
"

# 数据相关的config
DATA_ARGS=" \
    --seq-len ${seq_len} \
    --packed-length 4096 \
    --micro-bsz ${micro_batch_size} \
    --micro-num ${gradient_accumulate_size} \
    --rampup-batch-size 1 \
    --fixed_random_dataset_seqlen True \
    --pack_sample_into_one False \
    --num-worker 4 \
"

# deepspeed相关的config
DEEPSPEED_ARGS="\
    --deepspeed \
    --deepspeed_config deepspeed_config.json \
"


torchrun --nproc_per_node 8 --nnodes $NNODES --node_rank $SLURM_PROCID --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
pretrain.py $MODEL_ARGS $DATA_ARGS $DEEPSPEED_ARGS

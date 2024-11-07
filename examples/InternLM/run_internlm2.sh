#!/bin/bash
set -x

# 指定从哪里导入deepspeed，通过无需安装的方式，可以直接使用自己修改的deepspeed
export PYTHONPATH=/mnt/petrelfs/xiongyingtong/XYT-DeepSpeed:$PYTHONPATH

# NCCL相关环境变量
# export NCCL_DEBUG=info
export NCCL_SOCKET_IFNAME="bond0"
export NCCL_IB_HCA="mlx5_2,mlx5_3,mlx5_4,mlx5_5"

# CUDA相关环境变量
# export CUDA_DEVICE_MAX_CONNECTIONS=1

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
micro_batch_size=2
gradient_accumulate_size=1

seq_len=4096
packed_len=$((${micro_batch_size} * ${seq_len}))

# InternLM2-7B
num_layers=32
hidden_size=4096
num_attention_heads=32
num_kv_attention_heads=8
vocab_size=92544
mlp_ratio=3.5
dtype=torch.bfloat16
model_type=internlm2


# InternLM2-70B
# num_layers=80
# hidden_size=8192
# num_attention_heads=64
# num_kv_attention_heads=8
# vocab_size=92544
# mlp_ratio=3.5
# dtype=torch.bfloat16
# model_type=internlm2



# 模型相关的config
MODEL_ARGS=" \
    --num-layers ${num_layers} \
    --hidden-size ${hidden_size} \
    --num-attention-heads ${num_attention_heads} \
    --num_kv_attention_heads ${num_kv_attention_heads}\
    --vocab-size ${vocab_size} \
    --mlp-ratio ${mlp_ratio} \
    --dtype ${dtype} \
    --model-type ${model_type} \
"

# 数据相关的config
DATA_ARGS=" \
    --seq-len ${seq_len} \
    --packed-length ${packed_len} \
    --micro-bsz ${micro_batch_size} \
    --micro-num ${gradient_accumulate_size} \
    --fixed-random-dataset-seqlen \
    --rampup-batch-size 1 \
    --num-worker 4 \
"


# deepspeed相关的config
DEEPSPEED_ARGS="\
    --deepspeed \
    --deepspeed_config deepspeed_config.json \
"


torchrun --nproc_per_node 8 --nnodes $NNODES --node_rank $SLURM_PROCID --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
pretrain_internlm2.py $MODEL_ARGS $DATA_ARGS $DEEPSPEED_ARGS

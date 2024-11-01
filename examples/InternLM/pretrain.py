import argparse
import os

import deepspeed
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F

from modeling_internlm import InternLM1
from modeling_internlm2 import InternLM2

from data.build_dataloader import get_train_dataloader
from data.process_data import get_batch_data
from utils import get_torch_profiler

def get_model(args, device):
    if args.model_type == "internlm1":
        model = InternLM1(
            num_layers=args.num_layers,
            hidden_size=args.hidden_size,
            num_attention_heads=args.num_attention_heads,
            vocab_size=args.vocab_size,
            mlp_ratio=args.mlp_ratio,
            dtype=torch.bfloat16 if args.dtype == 'torch.bfloat16' else torch.float32,
            parallel_output=args.parallel_output,
            device=device,
        )
    else:
        model = InternLM2(
            num_layers=args.num_layers,
            hidden_size=args.hidden_size,
            num_attention_heads=args.num_attention_heads,
            num_kv_attention_heads=args.num_kv_attention_heads,
            vocab_size=args.vocab_size,
            mlp_ratio=args.mlp_ratio,
            no_bias=True,
            first=True,
            last=True,
            dtype=torch.bfloat16 if args.dtype == 'torch.bfloat16' else torch.float32,
            parallel_output=args.parallel_output,
            device=device,
        )
    return model

def print_log(msg):
    global_rank = dist.get_rank()
    if global_rank == 0:
        print(msg, flush=True)

def pretrain(args):
    
    # initialize deepspeed
    deepspeed.init_distributed()
    
    local_rank = int(os.environ["LOCAL_RANK"])  # 获取本地的rank
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # initialize model
    print_log(f">>building model {args.model_type}...")
    model = get_model(args, device)
    print_log(">>after building model...")
    print_log(model)
    # print(f"xyt debug after building model device = {device}", flush=True)

    print_log(">>initialize deepspeed...")
    model, _, _, _ = deepspeed.initialize(
                        args=args,
                        model=model,
                        model_parameters=model.parameters(),
                        dist_init_required=True)
    print_log(">>after initialize deepspeed...")
    # print(f"xyt debug model device = {model.device}", flush=True)

    # initialize dataloader
    print_log(">>initialize dataloader...")
    data_loader = get_train_dataloader(args)
    print_log(">>after initialize dataloader...")
    
    # initialize Loss function
    print_log(">>initialize loss...")
    loss_fn = nn.CrossEntropyLoss(
        ignore_index=-100,
        reduction="mean",
        label_smoothing=0,
    )
    print_log(">>after initialize loss...")

    llm_profile = get_torch_profiler(args.profiling)

    with llm_profile as prof:
        for idx, batch in enumerate(data_loader):
            
            input_ids, labels, kwargs = get_batch_data(batch, model.device)
            
            # forward
            logits = model(input_ids=input_ids, **kwargs)
            
            # compute loss
            shift_logits = logits.contiguous().view(-1, logits.size(-1))
            shift_labels = labels.contiguous().view(-1)
            loss = loss_fn(shift_logits, shift_labels)

            # backward
            model.backward(loss)

            # update
            model.step()
            
            print_log(f"step = {idx}, {loss=}")
            
            if idx % 3 == 0:
                prof.step()
        


def parse_args():
    parser = argparse.ArgumentParser(description='DeepSpeed InternLM Training')
    
    group_model = parser.add_argument_group(title='model')
    group_model.add_argument('--num-layers', type=int, default=32,
                             help="The number of layer for InternLM")
    group_model.add_argument('--hidden-size', type=int, default=4096)
    group_model.add_argument('--num-attention-heads', type=int, default=32)
    group_model.add_argument('--num_kv_attention_heads', type=int, default=8)
    group_model.add_argument('--vocab-size', type=int, default=50304)
    group_model.add_argument('--mlp-ratio', type=str, default=None)
    group_model.add_argument('--parallel-output', type=bool, default=False)
    group_model.add_argument('--dtype', type=str, default='torch.bfloat16')
    group_model.add_argument('--profiling', type=bool, default=False)
    group_model.add_argument('--model-type', type=str, default='internlm2')
    
    
    group_data = parser.add_argument_group(title='data')
    group_data.add_argument('--seq-len', type=int, default=4096)
    group_data.add_argument('--packed-length', type=int, default=4096)
    group_data.add_argument('--micro-bsz', type=int, default=1)
    group_data.add_argument('--micro-num', type=int, default=1)
    group_data.add_argument('--rampup-batch-size', type=int, default=1)
    group_data.add_argument('--fixed_random_dataset_seqlen', type=bool, default=True)
    group_data.add_argument('--pack_sample_into_one', type=bool, default=True)
    group_data.add_argument('--num-worker', type=int, default=4)
    
    parser = deepspeed.add_config_arguments(parser)
    
    args = parser.parse_args()
    
    args.mlp_ratio = eval(args.mlp_ratio)
    
    return args

if __name__ == "__main__":
    
    args = parse_args()
    pretrain(args)
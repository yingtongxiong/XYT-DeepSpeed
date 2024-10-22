import argparse

import deepspeed
import torch
import torch.distributed as dist

from modeling_internlm import InternLM1
from examples.data.build_dataloader import get_train_dataloader

def print_log(msg):
    global_rank = dist.get_rank()
    if global_rank == 0:
        print(msg, flush=True)

def pretrain(args):
    
    # initialize deepspeed
    deepspeed.init_distributed()

    # # initialize model
    # print_log(">>building model...")
    # model = InternLM1(
    #     num_layers=args.num_layers,
    #     hidden_size=args.hidden_size,
    #     num_attention_heads=args.num_attention_heads,
    #     vocab_size=args.vocab_size,
    #     mlp_ratio=args.mlp_ratio,
    #     dtype=torch.bfloat16 if args.dtype == 'torch.bfloat16' else torch.float32,
    #     parallel_output=args.parallel_output,
    #     device=torch.cuda.current_device(),
    # )
    # print_log(">>after buildin model...")
    
    # print_log(">>initialize deepspeed...")
    # model, _, _, _ = deepspeed.initialize(
    #                     args=args,
    #                     model=model,
    #                     model_parameters=model.parameters(),
    #                     dist_init_required=True)
    # print_log(">>after initialize deepspeed...")
    
    
    # initialize dataloader
    print_log(">>initialize dataloader...")
    data_loader = get_train_dataloader(args)
    print_log(">>after initialize dataloader...")
    
    # for _, batch in enumerate(data_loader):
    #     inputs, labels = batch[0].to(model.device), batch[1].to(model.device)
    #     # forward
    #     logits = model(inputs)
    #     # loss
    #     loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-1)
    #     # backward
    #     model.backward(loss)
    #     # update
    #     model.step()


def parse_args():
    parser = argparse.ArgumentParser(description='DeepSpeed InternLM Training')
    
    group_model = parser.add_argument_group(title='model')
    group_model.add_argument('--num-layers', type=int, default=32,
                             help="The number of layer for InternLM")
    group_model.add_argument('--hidden-size', type=int, default=4096)
    group_model.add_argument('--num-attention-heads', type=int, default=32)
    group_model.add_argument('--vocab-size', type=int, default=50304)
    group_model.add_argument('--mlp-ratio', type=float, default=2.5)
    group_model.add_argument('--parallel-output', type=bool, default=False)
    group_model.add_argument('--dtype', type=str, default='torch.bfloat16')
    
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
    return args

if __name__ == "__main__":
    
    args = parse_args()
    pretrain(args)
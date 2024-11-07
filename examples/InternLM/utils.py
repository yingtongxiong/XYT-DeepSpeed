import torch

class DummyProfile:
    """
    Dummy Profile.
    """

    def __init__(self, *args, **kwargs) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, a, b, c):
        pass

    def step(self):
        pass


def get_torch_profiler(is_profiling):
    if is_profiling:
        if torch.distributed.get_rank() == 0:
            schedule_config = {"wait": 1, "warmup": 1, "active": 1, "repeat": 1, "skip_first": 3}
            trace_path = "output/ds_logs/"
            llm_profile = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(**schedule_config),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(trace_path),
                with_stack=True,
                with_modules=True,
                profile_memory=True,
            )
        else:
            llm_profile = DummyProfile()
    else:
        llm_profile = DummyProfile()
    
    return llm_profile

def _split(input_, group, dim=-1):
    # skip if only one rank involved
    world_size = torch.distributed.get_world_size(group)
    if world_size == 1:
        return input_

    # Split along last dimension.
    dim_size = input_.size(dim)
    assert dim_size % world_size == 0, (
        f"The dimension to split ({dim_size}) is not a multiple of world size ({world_size}), "
        f"cannot split tensor evenly"
    )

    tensor_list = torch.split(input_, dim_size // world_size, dim=dim)
    rank = torch.distributed.get_rank(group)
    output = tensor_list[rank].contiguous()
    output = output.detach().clone()

    return output


def _gather(input_, group, dim=-1):
    # skip if only one rank involved
    world_size = torch.distributed.get_world_size(group)
    if world_size == 1:
        return input_

    # all gather
    rank = torch.distributed.get_rank(group)
    tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    tensor_list[rank] = input_
    torch.distributed.all_gather(tensor_list, input_, group=group)

    # concat
    output = torch.cat(tensor_list, dim=dim).contiguous()

    return output


class _GatherForwardSplitBackward(torch.autograd.Function):
    """Gather the input from model parallel region and concatenate.

    Args:
        input_: input matrix.
        parallel_mode: parallel mode.
        dim: dimension
    """

    @staticmethod
    def symbolic(input_):
        return _gather(input_, group=None)

    @staticmethod
    def forward(ctx, input_, group, dim):
        ctx.group = group
        ctx.dim = dim
        return _gather(input_, group, dim)

    @staticmethod
    def backward(ctx, grad_output):
        return _split(grad_output, ctx.group, ctx.dim), None, None


def gather_forward_split_backward(input_, group, dim):
    return _GatherForwardSplitBackward.apply(input_, group, dim)
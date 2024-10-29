# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import types
from deepspeed.utils import get_full_hp_param, get_full_hp_grad, get_hp_fragment_mapping
from deepspeed.utils import set_full_hp_param, set_full_hp_grad


def link_hp_params(lp_param_list, flat_hp_partition, gradient_dict, offload_gradient_dict, use_offload,
                   param_group_index, partition_start, partition_size, dp_group):
    # 获取当前partition中包含的param在flat_hp_partition的起始位置
    local_lp_param_and_offset = _init_lp_to_hp_mapping(lp_param_list, partition_start, partition_size, dp_group)

    # 获取完起始位置后，变求解被partition选中的lp_param，并记录被选中的部分分别在lp和hp的local索引
    for lp_param, lp_start in local_lp_param_and_offset:
        lp_param._hp_mapping = get_hp_fragment_mapping(lp_param, lp_start, flat_hp_partition, gradient_dict,
                                                       offload_gradient_dict, use_offload, param_group_index,
                                                       partition_start, partition_size)


def lazy_init_hp_params_optimizer_state(lp_param_list, flat_hp_partition, optimizer_state):
    for lp in lp_param_list:
        # 只有属于自己partition的param才有_hp_mapping
        if lp._hp_mapping is not None:
            lp._hp_mapping.set_optim_state_fragment(flat_hp_partition, optimizer_state[flat_hp_partition])


def _init_lp_to_hp_mapping(lp_param_list, partition_start, partition_size, dp_group):
    current_offset = 0
    param_and_offset_list = []
    partition_end = partition_start + partition_size
    index_in_param_group = 0
    for i, lp_param in enumerate(lp_param_list):
        lp_param._hp_mapping = None
        lp_param._dp_group = dp_group
        # get_full_hp_param的self会自动传入lp_param
        lp_param.get_full_hp_param = types.MethodType(get_full_hp_param, lp_param)
        lp_param.get_full_hp_grad = types.MethodType(get_full_hp_grad, lp_param)
        lp_param.set_full_hp_param = types.MethodType(set_full_hp_param, lp_param)
        lp_param.set_full_hp_grad = types.MethodType(set_full_hp_grad, lp_param)

        # lp_param overlaps with partition if both are true
        # 1) current_offset < partition_end,
        # 2) current_offset + lp_param.numel() >= partition_start
        lp_param_end = current_offset + lp_param.numel()
        # 表示如果lp_param的某一部分在区间[partition_start, partition_end]中，则记录当前lp_param在flatten中的开始索引
        # 即current_index表示了lp_param在flat_hp_partition中的起始位置
        if current_offset < partition_end and lp_param_end > partition_start:
            param_and_offset_list.append((lp_param, current_offset))
            lp_param._index_in_param_group = index_in_param_group
            # Indices for params in this partition/GPU
            index_in_param_group += 1
        current_offset += lp_param.numel()

    return param_and_offset_list

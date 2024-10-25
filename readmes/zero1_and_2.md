optimizer data sturctures:

``self.bit16_groups``: 长度为param group的个数，其中每个元素为某个param group中                 requires_grad为True的参数list

[[param0, param1,...], [param2, param3, ...], ...]

``self.round_robin_bit16_groups``: 长度为param group的个数
当self.round_robin_gradients=False，且和self.bit16_groups共享存储：
[[param0, param1,...], [param2, param3, ...], ...]

``self.round_robin_bit16_indices``: 长度为param group的个数，其中每个元素为参数组中每个参数的local索引
[[0, 1, ...], [0, 1, ...], ...]

``self.round_robin_bit16_meta``: 长度为param group的个数
[[meta_param0, meta_param1,...], [meta_param2, meta_param3, ...], ...]

其中 meta_param为并未实际创建的张量，其形状和数据类型与param相同，可以理解为只存储了param的形状和数据类型

``self.bit16_groups_flat``: 保存self.round_robin_bit16_groups flatten之后的参数
[flatten_params0, flatten_params1, ...]

``self.parallel_partitioned_bit16_groups``: 保存每个group的分组情况
[[dp0_param_group0, dp1_param_group0, ...], [dp_0_param_group1, dp_1_param_group1, ...], ... ]

``self.single_partition_of_fp32_groups``: 保存每个dp rank在每个param group被分配到的params

[[dp0_param_group0], [dp_0_param_group1], ...]
single_grad_partition = torch.zeros(int(self.partition_size[i]),dtype=self.single_partition_of_fp32_groups[i].dtype,device=self.device)
self.single_partition_of_fp32_groups[i].grad = single_grad_partition

``self.param_id``: key: id(param); value: count(所有param group中的索引)

``self.param_dict``: key: count; value: param

``self.params_in_partition``：长度为param group个数

[[param0, param1,...], [param1, param2, ...], ...]

``self.params_not_in_partition``：长度为param group个数
[[param0, param1,...], [param1, param2, ...], ...]

``self.first_offset``: 长度为param group个数
[int0, int1, ...]

由于参数是flatten之后被切的，因此有些参数，可能被分成多个部分，且多个部分被分配给不同的dp,因此需要用first_offset来记录当前的partition是在params_in_partition[0]中的local索引

``self.is_param_in_current_partition``: key: 在所有参数（包括所有参数组）中的索引；value：Bool 表示该param是否属于当前partition

``self.partition_size``: 长度为param group个数， 保存每个partition被分到的flatten后的参数的size
[int0, int1, ...]

``self.param_to_partition_ids``: 表示param属于哪个partition
self.param_to_partition_ids[param_group_id][param_id] = partition_id
其中，param_group_id表示group索引，param_id表示在所有参数中的全局索引
注意这里的param是没有被flatten的

``self.total_grads_in_partition``：表示该partition的grad总数
self.total_grads_in_partition[param_group_id][partition_id] = grad_count

``self.grad_start_offset``: 保存的是first_offset
self.grad_start_offset[param_group_id][partition_id][param_id] = first_offset
如果该param不属于当前partition，则没有param_id这个key

``self.is_grad_computed``: 判断当前grad是否被compute
self.is_grad_computed[param_group_id][partition_id][param_id] = bool

``self.grad_partition_insertion_offset``: ???

``self.param_names``: key: param; value: param_name

``self._param_slice_mappings``: 保存每个param被partition选中的部分在flat_hp_partition中的local索引，长度为param group的个数

[{"param_name": hp_fragment_address}, {}, ...]


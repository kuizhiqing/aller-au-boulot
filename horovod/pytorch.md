# PyTorch

## API

`horovod.torch` api

大部分 api 从 mpi_ops 中引入

```python
# horovod/torch/__init__.py

from horovod.torch import elastic
from horovod.torch.compression import Compression
from horovod.torch.functions import allgather_object, broadcast_object, broadcast_optimizer_state, broadcast_parameters
from horovod.torch.mpi_ops import allreduce, allreduce_async, allreduce_, allreduce_async_
from horovod.torch.mpi_ops import grouped_allreduce, grouped_allreduce_async, grouped_allreduce_, grouped_allreduce_async_
from horovod.torch.mpi_ops import sparse_allreduce_async
from horovod.torch.mpi_ops import allgather, allgather_async
from horovod.torch.mpi_ops import broadcast, broadcast_async, broadcast_, broadcast_async_
from horovod.torch.mpi_ops import alltoall, alltoall_async
from horovod.torch.mpi_ops import reducescatter, reducescatter_async
from horovod.torch.mpi_ops import join
from horovod.torch.mpi_ops import barrier
from horovod.torch.mpi_ops import poll, synchronize
from horovod.torch.mpi_ops import init, shutdown
from horovod.torch.mpi_ops import is_initialized, start_timeline, stop_timeline
from horovod.torch.mpi_ops import size, local_size, cross_size, rank, local_rank, cross_rank
from horovod.torch.mpi_ops import mpi_threads_supported, mpi_enabled, mpi_built
from horovod.torch.mpi_ops import gloo_enabled, gloo_built
from horovod.torch.mpi_ops import nccl_built, ddl_built, ccl_built, cuda_built, rocm_built
from horovod.torch.mpi_ops import ProcessSet, global_process_set, add_process_set, remove_process_set
from horovod.torch.mpi_ops import Average, Sum, Adasum
from horovod.torch.optimizer import DistributedOptimizer
from horovod.torch.sync_batch_norm import SyncBatchNorm
```

`mpi_ops`

这里的 api 分为两部分

* 一部分从 mpi_lib_v2 library 中通过 C api 暴露，然后通过 basic 引入
* 通信 api 通过 pybind 调用

```python
# horovod/torch/mpi_ops.py

# so library
from horovod.torch import mpi_lib_v2 as mpi_lib

_basics = _HorovodBasics(__file__, 'mpi_lib_v2')
# import basic methods
# mpi_ops 中会包含 basic 中的 api

# 重要
# handle 会被放在 map 中，避免被 gc
# 在 synchronize 之后被释放
_handle_map = {}

# inplace allreduce
# allreduce_ -> allreduce_async_ + synchronize -> _allreduce_async -> mpi_lib.horovod_torch_allreduce_async_

# allreduce
# allreduce -> HorovodAllreduce.apply -> allreduce_async + synchronize -> _allreduce_async -> mpi_lib.horovod_torch_allreduce_async_

def _allreduce_function_factory(tensor):
    return 'horovod_torch_allreduce_async_' + tensor.type().replace('.', '_')


def _allreduce_async(tensor, output, name, op, prescale_factor, postscale_factor, process_set: ProcessSet):
    function = _check_function(_allreduce_function_factory, tensor)
    try:
        handle = getattr(mpi_lib, function)(tensor, output, divisor,
                                            name.encode() if name is not None else _NULL, op,
                                            prescale_factor, postscale_factor, process_set.process_set_id)
    except RuntimeError as e:
        raise HorovodInternalError(e)
    return handle


def allreduce_async(tensor, average=None, name=None, op=None,
                    prescale_factor=1.0, postscale_factor=1.0,
                    process_set=global_process_set):
    output = tensor.new(tensor.shape)
    return _allreduce_async(tensor, output, name, op, prescale_factor, postscale_factor, process_set)


class HorovodAllreduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, average, name, op, prescale_factor, postscale_factor, process_set):
        ctx.average = average
        ctx.op = op
        ctx.prescale_factor = prescale_factor
        ctx.postscale_factor = postscale_factor
        ctx.process_set = process_set
        handle = allreduce_async(tensor, average, name, op, prescale_factor, postscale_factor, process_set)
        return synchronize(handle)

    @staticmethod
    def backward(ctx, grad_output):
        return allreduce(grad_output, average=ctx.average, op=ctx.op,
                         prescale_factor=ctx.prescale_factor,
                         postscale_factor=ctx.postscale_factor,
                         process_set=ctx.process_set), None, None, None, None, None, None


def allreduce(tensor, average=None, name=None, compression=Compression.none, op=None,
              prescale_factor=1.0, postscale_factor=1.0, process_set=global_process_set):
    """
    This acts as a thin wrapper around an autograd function.  If your input
    tensor requires gradients, then callings this function will allow gradients
    to be computed and backpropagated.
    """
    tensor_compressed, ctx = compression.compress(tensor)
    summed_tensor_compressed = HorovodAllreduce.apply(tensor_compressed, average, name, op,
                                                      prescale_factor, postscale_factor,
                                                      process_set)
    return compression.decompress(summed_tensor_compressed, ctx)


def allreduce_async_(tensor, average=None, name=None, op=None,
                     prescale_factor=1.0, postscale_factor=1.0,
                     process_set=global_process_set):
    op = handle_average_backwards_compatibility(op, average)
    return _allreduce_async(tensor, tensor, name, op, prescale_factor, postscale_factor, process_set)


def allreduce_(tensor, average=None, name=None, op=None,
               prescale_factor=1.0, postscale_factor=1.0,
               process_set=global_process_set):
    handle = allreduce_async_(tensor, average, name, op, prescale_factor, postscale_factor, process_set)
    return synchronize(handle)

```

这里最终调用的例如 `mpi_lib.horovod_torch_allreduce_async_` api 来自 cpp api 的 pybind.

cpp api

```cpp
// horovod/torch/mpi_ops_v2.cc

PYBIND11_MODULE(mpi_lib_v2, m) {
  m.def("horovod_torch_allreduce_async_torch_IntTensor", &DoAllreduce);
  ...
}

int DoAllreduce(::torch::Tensor tensor, ::torch::Tensor output, int divisor,
                const std::string& name, int reduce_op_int,
                double prescale_factor, double postscale_factor,
                int process_set_id) {
  auto handle = handle_manager.AllocateHandle();
  common::ReadyEventList ready_event_list;
  auto hvd_tensor = std::make_shared<TorchTensor>(tensor);
  auto hvd_context = std::make_shared<TorchOpContext>(device, output);
  auto hvd_output = std::make_shared<TorchTensor>(output);

  ReduceOp reduce_op = static_cast<ReduceOp>(reduce_op_int);
  auto enqueue_result = EnqueueTensorAllreduce(hvd_context, hvd_tensor, hvd_output, ready_event_list, ... handle);

  return handle;
}
```

这里主要包括两个内容

* torch tensor 到 horovod tensor 的转换
* 调用来自 horovod/common/operations.h 的 `EnqueueTensorAllreduce` 将计划通信的 tensor 加入队列

### HandleManager 

HandleManager 用于记录通信操作的返回结果 Status, 通过它可以知道操作完成情况和进行同步。

```cpp
// horovod/torch/handle_manager.h

class HandleManager {
  std::unordered_map<int, std::shared_ptr<Status>> results_;
};
```

### TorchTensor

TorchTensor 是 horovod tensor 的子类，它实现了对应的接口，horovod 的通信中可以直接使用该对象.

```cpp
// horovod/torch/adapter_v2.h

class TorchPersistentBuffer : public PersistentBuffer {
  AccessData(std::shared_ptr<OpContext> context) const override;
  ::torch::Tensor tensor_;
};

class TorchTensor : public Tensor {
  ::torch::Tensor tensor_;
};

class TorchOpContext : public OpContext {
  std::vector<::torch::Tensor> outputs_;
};
```

## Develop

总结 PyTorch 接入 Horovod 框架需要做的工作

* API 封装，暴露拓展的 API 给用户，可以使用 torch 的 tensor 作为参数传递等；在 `horovod/torch/mpi_ops.py` 中实现
* API 对接，前端 API 调用 horovod API 以及绑定 py API；在 `horovod/torch/mpi_ops_v2.cc` 中实现
* Tensor 适配，即 torch 的 tensor 能够在 horovord 框架中使用；在 `horovod/torch/adapter_v2.*` 中实现

## DistributedOptimizer

### Demo

```python
output = model(data)
loss = F.nll_loss(output, target)
loss.backward()
optimizer.synchronize()
torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
with optimizer.skip_synchronize():
    optimizer.step()
```

```python
class _DistributedOptimizer(torch.optim.Optimizer):
    def __init__(self, ...):
        self._register_hooks()

    def _register_hooks(self):
        for param_group in self.param_groups:
            for p in param_group['params']:
                if p.requires_grad:
                    self._requires_update.add(p)
                    # p_tmp 和 p 使用同样的 storage，不占用额外显存
                    p_tmp = p.expand_as(p)
                    grad_acc = p_tmp.grad_fn.next_functions[0][0]
                    grad_acc.register_hook(self._make_hook(p))
                    self._grad_accs.append(grad_acc)

    def _make_hook(self, p):
        def hook(*ignore):
            handle, ctx = None, None
            self._allreduce_delay[p] -= 1
            if self._allreduce_delay[p] == 0:
                handle, ctx = self._allreduce_grad_async(p)
            self._handles[p] = (handle, ctx)
        return hook

    def _allreduce_grad_async(self, p):
        name = self._parameter_names.get(p)
        tensor = p.grad

        if p.grad.is_sparse:
            if self.sparse_as_dense:
                tensor = tensor.to_dense()
            else:
                return self._sparse_allreduce_grad_async(p, name)

        tensor_compressed, ctx = self._compression.compress(tensor)

        handle = allreduce_async_(tensor_compressed, name=name, op=self.op,
                                  prescale_factor=prescale_factor,
                                  postscale_factor=postscale_factor,
                                  process_set=self.process_set)
        return handle, ctx

    def _sparse_allreduce_grad_async(self, p, name):
        handle = sparse_allreduce_async(p.grad, name=name, op=self.op,
                                        process_set=self.process_set)
        return handle, None

    def synchronize(self):
        if not self.process_set.included():
            self._synchronized = True
            return

        completed = set()
        for x in self._handles.keys():
          completed.update(x) if isinstance(x, tuple) else completed.add(x)
        missing_p = self._requires_update - completed

        # 处理 hook 注册不成功，或者说 hook 没有被调用
        # hook 被调用后会添加 self._handles
        for p in missing_p:
            handle, ctx = self._allreduce_grad_async(p)
            self._handles[p] = (handle, ctx)

        # handle 为 None 的 hook 跳过又不跳过了？
        # 需要注意 synchronize 函数每个 step 被调用，但不是每次 backward 都会被调用
        # 在之前的 train 中有每个 step 会多次 backward，所以 grad 的 hook 会被多次调用，次数匹配
        # 所以代码执行到这里 handle 应该是一次调用_allreduce_grad_async 如果不是就补上
        for p, (handle, ctx) in self._handles.items():
            if handle is None:
                handle, ctx = self._allreduce_grad_async(p)
                self._handles[p] = (handle, ctx)

        # for 循环处理异步通信的结果
        for p, (handle, ctx) in self._handles.items():
                # When handle is a callable function, it returns the aggregated tensor result
                output = synchronize(handle) if not callable(handle) else handle()
                self._allreduce_delay[p] = self.backward_passes_per_step
                if p.grad.is_sparse:
                    aggregated = self._compression.decompress(output, ctx)
                    if not aggregated.is_sparse:
                        aggregated = aggregated.to_sparse()

                    # Sparse grads do not support set_ for some reason, so we do this as an equivalent
                    p.grad.zero_().add_(aggregated)
                else:
                    p.grad.set_(self._compression.decompress(output, ctx))
        self._handles.clear()

        self._synchronized = True


def DistributedOptimizer(optimizer, named_parameters=None,
                         compression=Compression.none,
                         backward_passes_per_step=1,
                         op=Average,
                         gradient_predivide_factor=1.0,
                         num_groups=0, groups=None,
                         sparse_as_dense=False,
                         process_set=global_process_set):
    if op != Adasum or size() == 1:
        cls = type(optimizer.__class__.__name__, (optimizer.__class__,),
                   dict(_DistributedOptimizer.__dict__))
        return cls(optimizer.param_groups, named_parameters, compression, backward_passes_per_step, op,
                   gradient_predivide_factor, groups, sparse_as_dense, process_set)
    else:
        cls = type(optimizer.__class__.__name__, (optimizer.__class__,),
                   dict(_DistributedAdasumOptimizer.__dict__))
        return cls(optimizer.param_groups, named_parameters, compression, backward_passes_per_step)
```

[Adasum](https://arxiv.org/abs/2006.02924) 是 allreduce 时一种计算合并 gradient 的方法，可以让结果更加稳定。


# torch api

```python
# horovod/torch/mpi_ops.py

_basics = _HorovodBasics(__file__, 'mpi_lib_v2')

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
  auto enqueue_result = EnqueueTensorAllreduce(hvd_context, hvd_tensor, hvd_output, ready_event_list, ...);

  return handle;
}
```

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

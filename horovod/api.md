# API

## HorovodBasics 

### Python API

* horovod 的基础 API，会被具体实现 (torch/tf) 使用
* 提供 C 接口的 py 封装，通过 ctypes 实现调用

```python
# horovod/common/basics.py

class HorovodBasics(object):
    def __init__(self, pkg_path, *args):
        # 加载 mpi lib 实现包
        self.MPI_LIB_CTYPES = ctypes.CDLL(full_path, mode=ctypes.RTLD_GLOBAL)

    def init(self, comm, process_sets):
        initialization_ok = self.MPI_LIB_CTYPES.horovod_init(...)
        # initialization_ok = self.MPI_LIB_CTYPES.horovod_init_multi_comm(...)

        _init_process_sets(process_sets)

    def shutdown(self):
    def is_initialized(self):
    def start_timeline(self, file_path, mark_cycles=False):
    def stop_timeline(self):
    def size(self):
    def local_size(self):
    def cross_size(self):
    def rank(self):
    def local_rank(self):
    def cross_rank(self):
    def is_homogeneous(self):
    def mpi_threads_supported(self):
    def mpi_enabled(self):
    def mpi_built(self):
    def gloo_enabled(self):
    def gloo_built(self):
    def nccl_built(self):
    def ddl_built(self):
    def ccl_built(self):
    def cuda_built(self):
    def rocm_built(self):
    def _add_process_set_impl(self, ranks: Sequence[int]) -> Optional[int]:
    def _remove_process_set_impl(self, process_set_id: int) -> Optional[int]:
    def _process_set_rank(self, process_set_id: int) -> int:
    def _process_set_size(self, process_set_id: int) -> int:
    def _get_process_set_ids_and_ranks(self) -> Dict[int, List[int]]:
    def _comm_process_set_id(self, comm: MPI.Comm) -> int:
```

### C API

这里的接口有两个部分

* 系统相关的 C 接口，通过 py 的 ctypes 引用
* 通信相关的接口，直接被调用

```c
// horovod/common/operations.h

namespace horovod {
namespace common {

extern "C" {

bool horovod_init(const int* ranks, int nranks, const int* process_set_ranks,
                  const int* process_set_sizes, int num_process_sets);

#if HAVE_MPI
// 使用 MPI communicators 初始化
bool horovod_init_multi_comm(MPI_Comm* comm, int ncomms,
                             const int* process_set_ranks_via_ranks,
                             const int* process_set_sizes_via_ranks,
                             int num_process_sets_via_ranks);
#endif

void horovod_shutdown();

int horovod_rank();
int horovod_local_rank();

int horovod_size();
int horovod_local_size();

// bool horovod_xxx_enabled();
// bool horovod_xxx_built();

int horovod_reduce_op_average();
int horovod_reduce_op_sum();
int horovod_reduce_op_adasum();

int horovod_add_process_set(const int *ranks, int nranks);
int horovod_remove_process_set(int process_set_id);
int horovod_process_set_rank(int process_set_id);
int horovod_process_set_size(int process_set_id);
int horovod_process_set_included(int process_set_id);
int horovod_number_of_process_sets();
void horovod_process_set_ids(int* ids_prealloc);
int horovod_process_set_ranks(int id, int* ranks_prealloc);

} // C API 结束

Status EnqueueTensorAllreduce(std::shared_ptr<OpContext> context,
                              std::shared_ptr<Tensor> tensor,
                              std::shared_ptr<Tensor> output,
                              ReadyEventList ready_event_list,
                              std::string name, int device,
                              StatusCallback callback,
                              ReduceOp reduce_op = ReduceOp::SUM,
                              double prescale_factor = 1.0,
                              double postscale_factor = 1.0,
                              int32_t process_set_id = 0);

Status EnqueueTensorAllreduces(std::vector<std::shared_ptr<OpContext>>& contexts,
                               std::vector<std::shared_ptr<Tensor>>& tensors,
                               std::vector<std::shared_ptr<Tensor>>& outputs,
                               std::vector<ReadyEventList>& ready_event_lists,
                               std::vector<std::string>& names,
                               int device,
                               std::vector<StatusCallback>& callbacks,
                               ReduceOp reduce_op = ReduceOp::SUM,
                               double prescale_factor = 1.0,
                               double postscale_factor = 1.0,
                               int32_t process_set_id = 0);

Status EnqueueTensorAllgather(std::shared_ptr<OpContext> context,
                              std::shared_ptr<Tensor> tensor,
                              ReadyEventList ready_event_list,
                              const std::string& name, int device,
                              StatusCallback callback,
                              int32_t process_set_id = 0);

Status EnqueueTensorBroadcast(std::shared_ptr<OpContext> context,
                              std::shared_ptr<Tensor> tensor,
                              std::shared_ptr<Tensor> output, int root_rank,
                              ReadyEventList ready_event_list,
                              const std::string& name, int device,
                              StatusCallback callback,
                              int32_t process_set_id = 0);

Status EnqueueTensorAlltoall(std::shared_ptr<OpContext> context,
                             std::shared_ptr<Tensor> tensor,
                             std::shared_ptr<Tensor> splits,
                             ReadyEventList ready_event_list,
                             const std::string& name, int device,
                             StatusCallback callback,
                             int32_t process_set_id = 0);

Status EnqueueTensorReducescatter(std::shared_ptr<OpContext> context,
                                  std::shared_ptr<Tensor> tensor,
                                  ReadyEventList ready_event_list,
                                  const std::string& name, int device,
                                  StatusCallback callback,
                                  ReduceOp reduce_op = ReduceOp::SUM,
                                  int32_t process_set_id = 0);

Status EnqueueJoin(std::shared_ptr<OpContext> context,
                   std::shared_ptr<Tensor> output_last_joined_rank,
                   ReadyEventList ready_event_list,
                   const std::string& name, int device,
                   StatusCallback callback,
                   int32_t process_set_id = 0);

Status EnqueueBarrier(StatusCallback callback,
                   int32_t process_set_id = 0);

} // namespace common
} // namespace horovod

#endif // HOROVOD_OPERATIONS_H
```

## PyTorch API

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
  auto enqueue_result = EnqueueTensorAllreduce(hvd_context, hvd_tensor, hvd_output, ready_event_list, ...);

  return handle;
}
```

* `EnqueueTensorAllreduce` 来自 horovod/common/operations.h

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

## Tensorflow api

### Demo

```python
    import tensorflow as tf
    import horovod.tensorflow as hvd


    # Initialize Horovod
    hvd.init()

    # Pin GPU to be used to process local rank (one GPU per process)
    config = tf.ConfigProto()
    config.gpu_options.visible_device_list = str(hvd.local_rank())

    # Build model...
    loss = ...
    opt = tf.train.AdagradOptimizer(0.01 * hvd.size())

    # Add Horovod Distributed Optimizer
    opt = hvd.DistributedOptimizer(opt)

    # Add hook to broadcast variables from rank 0 to all other processes during
    # initialization.
    hooks = [hvd.BroadcastGlobalVariablesHook(0)]

    # Make training operation
    train_op = opt.minimize(loss)

    # Save checkpoints only on worker 0 to prevent other workers from corrupting them.
    checkpoint_dir = '/tmp/train_logs' if hvd.rank() == 0 else None

    # The MonitoredTrainingSession takes care of session initialization,
    # restoring from a checkpoint, saving to a checkpoint, and closing when done
    # or an error occurs.
    with tf.train.MonitoredTrainingSession(checkpoint_dir=checkpoint_dir,
                                           config=config,
                                           hooks=hooks) as mon_sess:
      while not mon_sess.should_stop():
        # Perform synchronous training.
        mon_sess.run(train_op)

```

```python
# horovod/tensorflow/mpi_ops.py 

def init(*args, **kwargs):
    _basics.init(*args, **kwargs)
    _setup_process_sets(_basics)

```


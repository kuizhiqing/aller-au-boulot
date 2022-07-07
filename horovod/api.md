# API

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

HorovodBasics 提供 mpi lib 的 c 接口封装
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

HorovodBasics 
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

}

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

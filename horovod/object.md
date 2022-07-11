# object

### Global State

全局状态类，用于保存全局信息，同时持有以下几个对象

```cpp
// horovod/common/global_state.h

struct HorovodGlobalState {
  // Background thread running MPI communication.
  std::thread background_thread;

  bool elastic_enabled = false;

  ParameterManager parameter_manager;

  // handles resizing and auto-tuning of buffer size.
  FusionBufferManager fusion_buffer;

  ProcessSetTable process_set_table;

  std::shared_ptr<Controller> global_controller;
};
```

### ParameterManager 

**WHAT**
ParameterManager 会记录 tunable 的变量，包括时间和 fusion buffer size 等，然后以获得高吞吐为目标对其进程动态调整

Manager 类主要包含几个部分

* manager 自身的配置包括开关、warmup 信息等等
* API 包括 update、tune 等
* 对可调整参数的封装，包括调整对象和调整方法
* 调整策略, 基于上述封装，实现怎样调整的策略

```cpp
// horovod/common/parameter_manager.h

class ParameterManager {
  // 根据 tensor 信息计算 score 和调用 Tune 调整参数
  // 返回值决定是否将更新后的参数广播出去
  bool Update(const std::vector<std::string>& tensor_names, int64_t bytes);
  // 根据 score 调整参数
  bool Tune(double score);

  // Manager 的配置
  struct Params {
    bool hierarchical_allreduce;
    bool hierarchical_allgather;
    bool cache_enabled;
    double tensor_fusion_threshold;
    double cycle_time;
    bool active;
  };
  Params GetParams();

  // Interface used to represent a parameter (or group of parameters) being tuned.
  class ITunableParameter {
  public:
    virtual bool Tune(double score, double* best_score) = 0;
    virtual void UpdateBestValue(double score) = 0;
    virtual double BestScore() const = 0;
    virtual bool IsTunable() const = 0;
  };

  // Abstract base class used to implement hierarchical parameter tuning.
  template <class T>
  class TunableParameter : public ITunableParameter {
    TunableParameter(T initial_value);
    T initial_value_;
    T value_;
    T best_value_;
  };
  // 需要调整的参数对象
  std::vector<ITunableParameter*> parameter_chain_;

  // A parameter that optimizes over a finite set of discrete values to be tried sequentially.
  template <class T>
  class CategoricalParameter : public TunableParameter<T> {
    CategoricalParameter(std::vector<T> values);
    std::vector<T> values_;
  };

  enum BayesianVariable { fusion_buffer_threshold_mb, cycle_time_ms };

  struct BayesianVariableConfig {
    BayesianVariable variable;
    std::pair<double, double> bounds;
  };

  // A set of numerical parameters optimized jointly using Bayesian Optimization.
  class BayesianParameter : public TunableParameter<Eigen::VectorXd> {
    std::unique_ptr<BayesianOptimization> bayes_;
  };

  CategoricalParameter<bool> hierarchical_allreduce_;
  CategoricalParameter<bool> hierarchical_allgather_;
  CategoricalParameter<bool> cache_enabled_;
  BayesianParameter joint_params_;
};
```

### FusionBufferManager

这里的 Buffer 会伴随进程的整个生命周期

```cpp
// horovod/common/fusion_buffer_manager.h

class FusionBufferManager {
public:
  Status InitializeBuffer(int64_t threshold,
                          int device, std::shared_ptr<OpContext> context,
                          int stream_id,
                          std::function<void()> on_start_init,
                          std::function<void()> on_end_init);
  // Status status = context->AllocatePersistent(threshold, &buffer);

  std::shared_ptr<PersistentBuffer> GetBuffer(int device, Framework framework, int stream_id);

  // map key: device ID, framework, stream_id
  std::unordered_map<std::tuple<int, Framework, int>,
      std::pair<std::shared_ptr<PersistentBuffer>, int64_t>> tensor_fusion_buffers_;
};

```

### ProcessSet & ProcessSetTable

基本调用流程

* HorovodGlobalState 中的 ProcessSetTable 对象
* ProcessSetTable Ids() 方法获取 ProcessSet 的 id
* ProcessSetTable Get() 方法从 id_to_process_set_ 变量中获取 ProcessSet 对象
* ProcessSet 的 IsCurrentProcessIncluded() 方法判断可见性

```cpp
// horovod/common/process_set.h

struct ProcessSet {
  std::shared_ptr<Controller> controller;

  TensorQueue tensor_queue;

  // LRU cache of Responses
  ResponseCache response_cache;

  // Information on registered groups.
  GroupTable group_table;

  std::vector<int> registered_global_ranks;

  MPIContext mpi_context;

  bool Initialize(const MPIContext& global_mpi_context);

  void Finalize(const Status& status);

  bool IsCurrentProcessIncluded() const;

  explicit ProcessSet(std::vector<int> global_ranks = {});
};

class ProcessSetTable {
  ProcessSetTable();

  void Initialize(const MPIContext& global_mpi_context);

  int32_t RegisterProcessSet(std::vector<int> global_ranks = {});

  std::vector<int32_t> Ids() const; // Returns copy to be threadsafe

  bool Contains(int32_t id) const;

  ProcessSet& Get(int32_t id);

  // Returns -1 if no process set with these ranks has been registered.
  int32_t FindId(const std::vector<int32_t>& ranks);

  void MarkProcessSetForRemoval(int32_t process_set_id);

  void DeregisterProcessSet(int32_t process_set_id);

  // ProcessSet 对象
  std::unordered_map<int32_t, ProcessSet> id_to_process_set_;
};
```

### Controller

```cpp
class Controller : public std::enable_shared_from_this<Controller> {
  void Initialize();

  virtual int GetTypeSize(DataType dtype) = 0;

  virtual void CrossRankBitwiseAnd(std::vector<long long>& bitvector, int count) = 0;

  virtual void CrossRankBitwiseOr(std::vector<long long>& bitvector, int count) = 0;

  virtual void Bcast(void* buffer, size_t size, int root_rank, Communicator communicator) = 0;

  virtual void AlltoallGetRecvSplits(const std::vector<int32_t>& splits, std::vector<int32_t>& recvsplits) = 0;

  virtual void Barrier(Communicator communicator) = 0;

  virtual void Allgather2Ints(std::array<int, 2> values, std::vector<int>& recv_values) = 0;

  void SynchronizeParameters();

  ResponseList ComputeResponseList(bool this_process_requested_shutdown, HorovodGlobalState& state, ProcessSet& process_set);

  virtual void DoInitialization() = 0;

  // For rank 0 to receive other ranks' ready tensors.
  virtual void RecvReadyTensors(std::vector<std::string>& ready_to_reduce,
                                std::vector<RequestList>& ready_list) = 0;

  // For other ranks to send their ready tensors to rank 0
  virtual void SendReadyTensors(RequestList& message_list) = 0;

  // For rank 0 to send final tensors ready to be allreduced/allgathered to other ranks.
  virtual void SendFinalTensors(ResponseList& response_list) = 0;

  // For other ranks to receive final ready tensors.
  virtual void RecvFinalTensors(ResponseList& response_list) = 0;

  // Once a tensor is ready to be reduced, the coordinator sends a Response
  // instructing all ranks to start the reduction to all ranks. The Response
  // also contains error messages in case the submitted Requests were not
  // valid (for example, contained mismatched shapes or types).
  // Constructing the Response, thus, requires a whole lot of error checking.
  Response ConstructResponse(const std::string& name, int joined_size = 0);

  // Routine to sync cache hit and invalid bit sets across workers.
  // Also determines global shutdown state and whether uncached requests
  // exist on any worker.
  void CoordinateCacheAndState(CacheCoordinator& cache_coordinator);

  void FuseResponses(std::deque<Response>& responses,
                     HorovodGlobalState& state,
                     ResponseList& response_list);

  // Return the total byte size of the final allgathered output tensor
  int64_t
  TotalByteSizeOfAllgatherOutput(const std::vector<int64_t>& tensor_sizes,
                                 const TensorTableEntry& entry);

  // Store the Request for a name, and return whether the total count of
  // Requests for that tensor is now equal to the HOROVOD size (and thus we are
  // ready to reduce the tensor).
  bool IncrementTensorCount(const Request& msg, int joined_size = 0);

  bool is_initialized_ = false;

  int rank_ = 0;
  int local_rank_ = 0;
  int cross_rank_ = 0;
  int size_ = 1;
  int local_size_ = 1;
  int cross_size_ = 1;
  bool is_coordinator_ = false;
  bool is_homogeneous_ = false;

  // Global rank of each process in the set associated to this controller.
  std::vector<int> global_ranks_;

  // Map (global rank) -> (process set controller rank) for each process in this
  // set.
  std::unordered_map<int,int> global_rank_to_controller_rank_;

  // Controller process set ranks of processes running on this node.
  std::vector<int> local_comm_ranks_;

  StallInspector stall_inspector_;

  // Only exists on the coordinator node (rank zero). Maintains a vector of
  // requests to allreduce every tensor (keyed by tensor name).
  MessageTable message_table_;

  // Outside dependencies
  TensorQueue& tensor_queue_;

  ResponseCache& response_cache_;

  ParameterManager& parameter_manager_;

  GroupTable& group_table_;
};
```

### GroupTable

在 ProcessSet 中持有，

* EnqueueTensorAllreduces 中调用 RegisterGroup
* RunLoopOnce 中 PerformOperation 前会调用 DeregisterGroups

```cpp
// horovod/common/group_table.h

class GroupTable {
public:
  GroupTable() = default;

  int32_t GetGroupIDFromTensorName(const std::string& tensor_name) const;
  const std::vector<std::string>& GetGroupTensorNames(int32_t group_id) const;
  bool empty(void) const;

  int32_t RegisterGroup(std::vector<std::string>&& tensor_names);
  void DeregisterGroups(const std::vector<std::string>& tensor_names);

  std::unordered_map<std::string, int32_t> tensor_name_to_id_;
  std::unordered_map<int32_t, std::vector<std::string>> id_to_tensor_names_;

  // Queue of ids that can be reused
  std::queue<int32_t> free_ids_;

  // Next available group id (increases each time a group is added)
  int32_t next_group_id_ = 0;
};
```

### OperationManager 

对通信 op 的一个封装，基本上通过调用 op->Excute 实现，为 horovod/common/ops 目录中的各个 op 实现提供了接口。

```cpp
// horovod/common/ops/operation_manager.h

class OperationManager {
  OperationManager(ParameterManager* param_manager,
                   std::vector<std::shared_ptr<AllreduceOp>> allreduce_ops,
                   std::vector<std::shared_ptr<AllgatherOp>> allgather_ops,
                   std::vector<std::shared_ptr<BroadcastOp>> broadcast_ops,
                   std::vector<std::shared_ptr<AlltoallOp>> alltoall_ops,
                   std::vector<std::shared_ptr<ReducescatterOp>> reducescatter_ops,
                   std::shared_ptr<JoinOp> join_op,
                   std::vector<std::shared_ptr<AllreduceOp>> adasum_ops,
                   std::shared_ptr<BarrierOp> barrier_op,
                   std::shared_ptr<ErrorOp> error_op);

  Status ExecuteAllreduce(std::vector<TensorTableEntry>& entries, const Response& response) const;

  Status ExecuteAllgather(std::vector<TensorTableEntry>& entries, const Response& response) const;

  Status ExecuteBroadcast(std::vector<TensorTableEntry>& entries, const Response& response) const;

  Status ExecuteAlltoall(std::vector<TensorTableEntry>& entries, const Response& response) const;

  Status ExecuteReducescatter(std::vector<TensorTableEntry>& entries, const Response& response) const;

  Status ExecuteError(std::vector<TensorTableEntry>& entries, const Response& response) const;

  Status ExecuteJoin(std::vector<TensorTableEntry>& entries,
                     const Response& response, ProcessSet& process_set) const;

  Status ExecuteAdasum(std::vector<TensorTableEntry>& entries, const Response& response) const;

  Status ExecuteBarrier(std::vector<TensorTableEntry>& entries, const Response& response) const;

  Status ExecuteOperation(std::vector<TensorTableEntry>& entries,
                          const Response& response,
                          ProcessSet& process_set) const;

  ParameterManager* param_manager_;

  std::vector<std::shared_ptr<AllreduceOp>> allreduce_ops_;
  std::vector<std::shared_ptr<AllgatherOp>> allgather_ops_;
  std::vector<std::shared_ptr<BroadcastOp>> broadcast_ops_;
  std::vector<std::shared_ptr<AlltoallOp>> alltoall_ops_;
  std::vector<std::shared_ptr<ReducescatterOp>> reducescatter_ops_;
  std::shared_ptr<JoinOp> join_op_;
  std::vector<std::shared_ptr<AllreduceOp>> adasum_ops_;
  std::shared_ptr<BarrierOp> barrier_op_;
  std::shared_ptr<ErrorOp> error_op_;
};
```

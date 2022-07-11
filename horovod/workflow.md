# workflow

### Operation

接口的具体实现，初始化会启动一个后台进程，不断执行通信操作

```c
// horovod/common/operations.cc

extern "C" {

bool horovod_init(const int* ranks, int nranks, const int* process_set_ranks,
                  const int* process_set_sizes, int num_process_sets) {
  return InitializeHorovodOnce(...);
}

bool horovod_init_multi_comm(MPI_Comm* comm, int ncomms,
                             const int* process_set_ranks_via_ranks,
                             const int* process_set_sizes_via_ranks,
                             int num_process_sets_via_ranks) {
  return InitializeHorovodOnce(std::vector<int>(), process_set_ranks_vecs);
}

// 启动 horovod 后台进程，只执行一次
bool InitializeHorovodOnce(
    const std::vector<int>& ranks,
    const std::vector<std::vector<int>>& process_set_ranks) {
  if (!horovod_global.initialize_flag.test_and_set()) {
    horovod_global.initialization_done = false;
    horovod_global.background_thread =
        std::thread(BackgroundThreadLoop, std::ref(horovod_global));
  }

  while (!horovod_global.initialization_done &&
         !horovod_global.initialization_failed) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

void BackgroundThreadLoop(HorovodGlobalState& state) {
  auto mpi_ctx_manager = MPIContextManager();
  if (global_mpi_context.IsEnabled()) {
    global_mpi_context.Initialize(mpi_ctx_manager);
    if (state.control_operation == LibType::MPI) {
      // Initializes global controller
      state.process_set_table.Initialize(global_mpi_context);
    }
  }

  bool is_coordinator = state.global_controller->IsCoordinator();
  bool is_homogeneous = state.global_controller->IsHomogeneous();
  int size = state.global_controller->GetSize();
  int local_size = state.global_controller->GetLocalSize();
  int local_rank = state.global_controller->GetLocalRank();

  # 一堆配置
  state.parameter_manager.SetTensorFusionThresholdBytes(128 * 1024 * 1024);
  state.parameter_manager.SetTensorFusionThresholdBytes(threshold, true);
  state.parameter_manager.SetCycleTimeMs(1);
  state.parameter_manager.SetCacheEnabled(true);
  state.process_set_table.Get(0).response_cache.set_capacity(...)
  state.parameter_manager.SetHierarchicalAllgather(false);
  state.parameter_manager.SetHierarchicalAllreduce(false);

  while (RunLoopOnce(state));

  state.shut_down = true;

  horovod_global.process_set_table.Finalize(global_mpi_context,...)
}

bool RunLoopOnce(HorovodGlobalState& state) {
  state.process_set_table.InitializeRegisteredAndRemoveMarkedIfReady(global_mpi_context);

  for (auto process_set_id : state.process_set_table.Ids()) {
    auto& process_set = state.process_set_table.Get(process_set_id);
    auto response_list = process_set.IsCurrentProcessIncluded()
            ? process_set.controller->ComputeResponseList(this_process_requested_shutdown, state, process_set)
            : ResponseList();

    if (process_set.IsCurrentProcessIncluded()) {
      int global_rank = state.global_controller->GetRank();
      for (auto& response : response_list.responses()) {
        PerformOperation(response, process_set);
      }
    }
  }
}
```

这里主要包含两个操作

* process_set.controller->ComputeResponseList
* PerformOperation

### PerformOperation

```cpp
// 执行通信操作，获取 Response 
void PerformOperation(Response response, ProcessSet& process_set) {
  std::vector<TensorTableEntry> entries;
  process_set.tensor_queue.GetTensorEntriesFromResponse(response, entries, process_set.joined);

  if (response.response_type() != Response::JOIN &&
      response.response_type() != Response::BARRIER) {
    if (entries.size() > 1) {
      auto first_entry = entries[0];
      // 创建 buffer
      Status status = horovod_global.fusion_buffer.InitializeBuffer(
          process_set.controller->TensorFusionThresholdBytes(),
          first_entry.device, first_entry.context,
          horovod_global.current_nccl_stream,
          [&]() { timeline.ActivityStartAll(entries, INIT_FUSION_BUFFER); },
          [&]() { timeline.ActivityEndAll(entries); });
    }
  }

  // std::unique_ptr<OperationManager> op_manager;
  Status status = op_manager->ExecuteOperation(entries, response, process_set);
}
```

*OperationManager->ExecuteOperation* 即调用对应 api 完成 op 的执行

### ComputeResponseList

这是 controller 里最重要的函数，它在 worker 间进行 allreduce/allgather 的协同，返回准备好通信的 tensor 列表，其中

* 0 号 worker 作为 coordinator
* 每个 worker 都存有一份别的 worker 发送的准备好的 tensor 列表作为 cache

具体流程如下

- worker 所有计划的通信操作都会先发送给 coordinator，包括 (tensor, reduce/gather, shape, type)
- worker 发送 DONE 消息给 coordinator 当所有计划通信操作都已发送
- coordinator 接受来自 worker 的计划通信请求，存在 request table 中，直到收集到所有节点的 DONE 消息
- coordinator 找到全部准备好的 tensor，向 worker 发送 Response 消息，当发送完毕时发送 DONE 消息
- worker 监听来自 coordinator 的消息，执行对应的 reduce/gather 操作，直到收到 DONE 消息

```cpp
// horovod/common/controller.cc

ResponseList Controller::ComputeResponseList(bool this_process_requested_shutdown,
                                             HorovodGlobalState& state,
                                             ProcessSet& process_set) {
  CacheCoordinator cache_coordinator(response_cache_.num_active_bits());
  std::deque<Request> message_queue_tmp;

  ResponseList response_list;

  if (!need_communication) {
    FuseResponses(responses, state, response_list);
  } else {
    std::vector<std::string> ready_to_reduce;

    if (is_coordinator_) {
      // Receive ready tensors from other ranks
      std::vector<RequestList> ready_list;
      RecvReadyTensors(ready_to_reduce, ready_list);

      // Process messages.
      for (int i = 1; i < size_; ++i) {
        auto received_message_list = ready_list[i];
        for (auto& received_message : received_message_list.requests()) {
          auto& received_name = received_message.tensor_name();
          ready_to_reduce.push_back(received_name);
        }
      }

      // 到此，coordinator 过程结束
      std::deque<Response> responses;

      for (auto& tensor_name : ready_to_reduce) {
        Response response = ConstructResponse(tensor_name, process_set.joined_size);
        responses.push_back(std::move(response));
      }
      FuseResponses(responses, state, response_list);

      // Broadcast final results to other ranks.
      SendFinalTensors(response_list);

    } else {
      RequestList message_list;
      while (!message_queue_tmp.empty()) {
        message_list.add_request(message_queue_tmp.front());
        message_queue_tmp.pop_front();
      }

      // Send ready tensors to rank zero
      SendReadyTensors(message_list);

      // Receive final tensors to be processed from rank zero
      RecvFinalTensors(response_list);
    }
  }

  return response_list;
}
```

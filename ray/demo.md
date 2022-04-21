
# Demo

```python
import ray
ray.init()
# ray.init(address='ray://localhost:10001')

@ray.remote
def f(x):
    return x * x

futures = [f.remote(i) for i in range(4)]
print(ray.get(futures))
```

```python
# python/ray/worker.py

def init(address: Optional[str] = None, ...):
    # if address
    builder = ray.client(address, _deprecation_warn_enabled=False)
    builder._init_args(**passed_kwargs)
    return builder.connect()

    # if bootstrap_address is None:
    _global_node = ray.node.Node(head=True, shutdown_at_exit=False, spawn_reaper=True, ray_params=ray_params)
    # else
    _global_node = ray.node.Node(ray_params, head=False, shutdown_at_exit=False, spawn_reaper=False, connect_only=True)

    connect(...)
    return RayContext(...)

def connect(node, worker=global_worker, ...):
    worker.node = node
    worker.core_worker = ray._raylet.CoreWorker(...)
```

CoreWorker
```cpp
// src/ray/core_worker/core_worker.h

class CoreWorker : public rpc::CoreWorkerServiceHandler {
  instrumented_io_context io_service_;
  boost::asio::io_service::work io_work_;

  rpc::CoreWorkerGrpcService grpc_service_;
  std::unique_ptr<rpc::GrpcServer> core_worker_server_;
  
  // std::unique_ptr<ObjectRecoveryManager> object_recovery_manager_;

  std::shared_ptr<TaskManager> task_manager_;
  std::unique_ptr<ActorManager> actor_manager_;
}


// src/ray/core_worker/core_worker.cc

CoreWorker::CoreWorker(...) {
  io_work_(io_service_),
  grpc_service_(io_service_, *this),

  core_worker_server_->RegisterService(grpc_service_);
  core_worker_server_->Run();
}

```



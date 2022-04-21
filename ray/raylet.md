# Raylet

服务由以下二进制启动
```shell
core/src/ray/raylet/raylet
```

程序入口
```cpp
// src/ray/raylet/main.cc

int main(int argc, char *argv[]) {

  // 启动 IO service
  // class instrumented_io_context : public boost::asio::io_context {...}
  instrumented_io_context main_service;
  boost::asio::io_service::work main_work(main_service);

  std::unique_ptr<ray::raylet::Raylet> raylet;
  ray::raylet::NodeManagerConfig node_manager_config;
  ray::ObjectManagerConfig object_manager_config;
  raylet = std::make_unique<ray::raylet::Raylet>(main_service,
                                                 raylet_socket_name,
                                                 node_ip_address,
                                                 node_manager_config,
                                                 object_manager_config,
                                                 gcs_client,
                                                 metrics_export_port);
  raylet->Start();
  main_service.run();
}

```

主服务类
```cpp
// src/ray/raylet/raylet.cc

class Raylet {
  // 用于和 gcs 连接的客户端
  std::shared_ptr<gcs::GcsClient> gcs_client_;
  NodeManager node_manager_;
}

void Raylet::Start() {
  RAY_CHECK_OK(RegisterGcs());

  // Start listening for clients.
  DoAccept();
}

ray::Status Raylet::RegisterGcs() {
  node_manager_.RegisterGcs();

  gcs_client_->Nodes().RegisterSelf(self_node_info_, register_callback);
}

void Raylet::DoAccept() {
  acceptor_.async_accept(
      socket_,
      boost::bind(&Raylet::HandleAccept, this, boost::asio::placeholders::error));
}

void Raylet::HandleAccept(const boost::system::error_code &error) {
  // 建立本地连接并分发到 node manager 处理
  auto new_connection = ClientConnection::Create(
      client_handler, // node_manager_.ProcessNewClient(client);
      message_handler, // node_manager_.ProcessClientMessage(client, message_type, message.data());
      std::move(socket_),
      "worker",
      node_manager_message_enum,
      static_cast<int64_t>(protocol::MessageType::DisconnectClient),
      message_data);
  // 处理连接
  DoAccept();
}
```

Node Manager

NodeManager 本身是一个 ServiceHandler，所以在初始化 node_manager_service_ 时，使用 this 作为 handler 传递。
```cpp
// src/ray/raylet/node_manager.h

class NodeManager : public rpc::NodeManagerServiceHandler {
  std::shared_ptr<gcs::GcsClient> gcs_client_;
  std::unique_ptr<HeartbeatSender> heartbeat_sender_;
  WorkerPool worker_pool_;
  ObjectManager object_manager_;
  rpc::GrpcServer node_manager_server_;
  rpc::NodeManagerGrpcService node_manager_service_;

  std::unique_ptr<rpc::AgentManagerServiceHandler> agent_manager_service_handler_;
  rpc::AgentManagerGrpcService agent_manager_service_;

  std::shared_ptr<ClusterResourceScheduler> cluster_resource_scheduler_;
  std::shared_ptr<LocalTaskManager> local_task_manager_;

  std::shared_ptr<PlacementGroupResourceManager> placement_group_resource_manager_;
}

// src/ray/raylet/node_manager.cc

// Push
// Pull

NodeManager::NodeManager(...) {
  // 非常多的初始化配置
  node_manager_service_(io_service, *this),
  // 然后注册服务并启动
  node_manager_server_.RegisterService(node_manager_service_);
  node_manager_server_.RegisterService(agent_manager_service_);
  node_manager_server_.Run();
}
```

NodeManager 的 rpc 接口
```cpp
// src/ray/rpc/node_manager/node_manager_server.h

// `NodeManagerService` 的接口, 对应 `src/ray/protobuf/node_manager.proto`.
class NodeManagerServiceHandler {}
// 目前有以下接口
// UpdateResourceUsage
// RequestResourceReport
// RequestWorkerLease
// ReportWorkerBacklog
// ReturnWorker
// ReleaseUnusedWorkers
// CancelWorkerLease
// PinObjectIDs
// GetNodeStats
// GlobalGC
// FormatGlobalMemoryInfo
// PrepareBundleResources
// CommitBundleResources
// CancelResourceReserve
// RequestObjectSpillage
// ReleaseUnusedBundles
// GetSystemConfig
// GetGcsServerAddress
// ShutdownRaylet

class NodeManagerGrpcService : public GrpcService {
  NodeManagerGrpcService(instrumented_io_context &io_service,
                         NodeManagerServiceHandler &service_handler)
}
```

> 关于怎么增加新的接口可以参考: `src/ray/core_worker/core_worker.h`.

ObjectManager 

```cpp
// src/ray/object_manager/object_manager.h

class ObjectManager : public ObjectManagerInterface, public rpc::ObjectManagerServiceHandler {
  instrumented_io_context rpc_service_;
  boost::asio::io_service::work rpc_work_;

  rpc::GrpcServer object_manager_server_;
  rpc::ObjectManagerGrpcService object_manager_service_;
}

// 主要接口
// Push
// Pull

// src/ray/object_manager/object_manager.cc

ObjectManager::ObjectManager(...){
  rpc_work_(rpc_service_),
  object_manager_server_("ObjectManager",...)
  object_manager_service_(rpc_service_, *this),
  StartRpcService();
}

void ObjectManager::StartRpcService() {
  // for i in config_.rpc_service_threads_number
  rpc_threads_[i] = std::thread(&ObjectManager::RunRpcService, this, i);
  object_manager_server_.RegisterService(object_manager_service_);
  object_manager_server_.Run();
}

void ObjectManager::RunRpcService(int index) {
  rpc_service_.run();
}
```

```cpp
// src/ray/rpc/object_manager/object_manager_server.h


class ObjectManagerGrpcService : public GrpcService {
  ObjectManagerGrpcService(instrumented_io_context &io_service,
                           ObjectManagerServiceHandler &service_handler)
      : GrpcService(io_service), service_handler_(service_handler){};
```

Worker Pool

```cpp
// src/ray/raylet/worker_pool.cc
```




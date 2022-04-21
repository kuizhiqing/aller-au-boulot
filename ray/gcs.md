# GCS Server

服务由以下二进制启动
```shell
"core/src/ray/gcs/gcs_server"
```

程序入口
```cpp
// src/ray/gcs/gcs_server/gcs_server_main.cc

int main(int argc, char *argv[]) {
  // 初始化配置
  RayConfig::instance().initialize(config_list);
  // 启动 IO service
  // class instrumented_io_context : public boost::asio::io_context {...}
  instrumented_io_context main_service;
  boost::asio::io_service::work work(main_service);
  // 初始化状态模块
  ray::stats::Init(global_tags, metrics_agent_port);

  // 启动 grpc 服务
  ray::gcs::GcsServerConfig gcs_server_config;
  gcs_server_config.grpc_server_name = "GcsServer";
  ray::gcs::GcsServer gcs_server(gcs_server_config, main_service);
  gcs_server.Start();

  main_service.run();
}
```

主服务
```cpp
// src/ray/gcs/gcs_server/gcs_server.cc

// 服务初始化，根据配置使用外置 redis 存储或者内置存储
GcsServer::GcsServer(...) {
  if (storage_type_ == "redis") {
    gcs_table_storage_ = std::make_shared<gcs::RedisGcsTableStorage>(GetOrConnectRedis());
  } else if (storage_type_ == "memory") {
    gcs_table_storage_ = std::make_shared<InMemoryGcsTableStorage>(main_service_);
  }
}

void GcsServer::Start() {
  // 异步加载 gcs tables 数据
  auto gcs_init_data = std::make_shared<GcsInitData>(gcs_table_storage_);
  gcs_init_data->AsyncLoad([this, gcs_init_data] { DoStart(*gcs_init_data); });
}

void GcsServer::DoStart(const GcsInitData &gcs_init_data) {
  // Init cluster resource scheduler.
  // Init gcs resource manager.
  // Init synchronization service
  // Init gcs node manager.
  // Init gcs heartbeat manager.
  // Init KV Manager
  // Init function manager
  // Init Pub/Sub handler
  // Init RuntimeENv manager
  // Init gcs job manager.
  // Init gcs placement group manager.
  // Init gcs actor manager.
  // Init gcs worker manager.
  // Init stats handler.
  // Install event listeners.

  // 启动 rpc 服务，依赖 tables 数据加载完成
  // rpc::GrpcServer rpc_server_;
  rpc_server_.Run();
  // 心跳服务
  gcs_heartbeat_manager_->Start();

  RecordMetrics();
}
```

Table storage
```cpp
// src/ray/gcs/gcs_server/gcs_table_storage.h

class GcsTable {}

class GcsTableWithJobId : public GcsTable<Key, Data> {}

class GcsTableStorage {};

class RedisGcsTableStorage : public GcsTableStorage {};

class InMemoryGcsTableStorage : public GcsTableStorage {};
```

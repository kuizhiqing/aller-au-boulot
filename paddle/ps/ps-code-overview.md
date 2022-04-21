# paddle ps 代码分析

@kuizhiqing

## python 前端

### API

```python
import paddle.distributed.fleet as fleet

fleet.init()

if fleet.is_server():
    fleet.init_server()
    fleet.run_server()
elif fleet.is_worker():
    run_worker()
    fleet.stop_worker()

def run_worker():
    # paddle.static.Executor(place)
    exe.run(paddle.static.default_startup_program())
    fleet.init_worker()
    exe.train_from_dataset(...)

# Fin, fleet is optimizer
```

### fleet init/optimizer

```python
# python/paddle/distributed/fleet/base/fleet_base.py

def init(self, role_maker=None, is_collective=False, strategy=None):
    # 配置之集大成者，就是各种配置，细到训练参数，粗到训练模式，开关
    strategy = DistributedStrategy()
    # role maker 包含分布式信息，基本上对接 launch 信息
    # 也负责初始化如 gloo 之类的工具
    self._role_maker._generate_role()

def minimize(...)
    def _minimize_impl(...)
        # runtime handle 做映射 init_server/_init_server, run_server/_run_server
        self._runtime_handle = RuntimeFactory()._create_runtime(context)
```

### runtime

```python
# 使用实例
# python/paddle/distributed/ps/the_one_ps.py
class TheOnePSRuntime(RuntimeBase):
    def __init__(self):
        super(TheOnePSRuntime, self).__init__()
        self._communicator = None
        self._server = None
        self._worker = fluid.core.DistFleetWrapper()
        self._server_sub_program = []
        self._heter_client = None
        self._send_ctx = None
```

### pybind

```cpp
void BindDistFleetWrapper(py::module* m) {
  py::class_<FleetWrapper, std::shared_ptr<FleetWrapper>>(*m,
                                                          "DistFleetWrapper")
      .def(py::init([]() { return FleetWrapper::GetInstance(); }))
      .def("load_sparse", &FleetWrapper::LoadSparseOnServer)
      .def("load_model", &FleetWrapper::LoadModel)
      .def("load_one_table", &FleetWrapper::LoadModelOneTable)
      .def("init_server", &FleetWrapper::InitServer)
      .def("run_server",
           (uint64_t (FleetWrapper::*)(void)) & FleetWrapper::RunServer)
      .def("run_server", (uint64_t (FleetWrapper::*)(          // NOLINT
                             const std::string&, uint32_t)) &  // NOLINT
                             FleetWrapper::RunServer)
      .def("init_worker", &FleetWrapper::InitWorker)
      .def("push_dense_params", &FleetWrapper::PushDenseParamSync)
      .def("pull_dense_params", &FleetWrapper::PullDenseVarsSync)
      .def("save_all_model", &FleetWrapper::SaveModel)
      .def("save_one_model", &FleetWrapper::SaveModelOneTable)
      .def("recv_and_save_model", &FleetWrapper::RecvAndSaveTable)
      .def("sparse_table_stat", &FleetWrapper::PrintTableStat)
      .def("stop_server", &FleetWrapper::StopServer)
      .def("stop_worker", &FleetWrapper::FinalizeWorker)
      .def("barrier", &FleetWrapper::BarrierWithTable)
      .def("shrink_sparse_table", &FleetWrapper::ShrinkSparseTable)
      .def("set_clients", &FleetWrapper::SetClients)
      .def("get_client_info", &FleetWrapper::GetClientsInfo)
      .def("create_client2client_connection",
           &FleetWrapper::CreateClient2ClientConnection);
}
```

## fleet run_server

```python
# runtime 层初始化
class TheOnePSRuntime(RuntimeBase):
    def _init_server(self, dirname=None, var_names=None, **kwargs):
        # cpp instance
        self._server = fluid.core.DistFleetWrapper()
        self._server.init_server(server_desc, self.string_hosts, role_id,
                                 trainers, self._server_sub_program)
        # load_sparse 
        for var_name in load_varnames:
            table_id = sparse_table_maps[var_name]
            self._server.load_sparse(dirname, "0", table_id)

    def _run_server(self):
        self._server.run_server(host, int(port))
```

### FleetWrapper

```cpp
// paddle/fluid/distributed/ps/wrapper/fleet.cc
// FleetWrapper 层
void FleetWrapper::InitServer(...){
    pserver_ptr_ = std::shared_ptr<paddle::distributed::PSCore>(
        new paddle::distributed::PSCore());
    pserver_ptr_->init_server(...)
}

uint64_t FleetWrapper::RunServer(...){
    auto ret = pserver_ptr_->run_server(ip, port);
}

void FleetWrapper::LoadSparseOnServer(...){
    // _server_ptr is PSServer
    pserver_ptr_->_server_ptr->table(table_id)->load(path, meta);
}
```

### PSCore

```cpp
// paddle/fluid/distributed/ps/service/ps_service/service.cc
// PSCore layer

int PSCore::init_server(...){
  _ps_env = paddle::distributed::PaddlePSEnvironment();
  _ps_env.set_ps_servers(host_sign_list, node_num);
  _ps_env.set_trainers(trainers); // 没啥用
  _server_ptr = std::shared_ptr<paddle::distributed::PSServer>(
      paddle::distributed::PSServerFactory::create(_ps_param));
  ret = _server_ptr->configure(_ps_param, _ps_env, index, server_sub_program);
}

uint64_t PSCore::run_server(const std::string& ip, uint32_t port) {
  return _server_ptr->start(ip, port);
}
```

### PSServer

```cpp
// paddle/fluid/distributed/ps/service/server.cc
// PSServer layer

PSServer *PSServerFactory::create(const PSParameter &ps_config) {
    PSServer *server =
      CREATE_PSCORE_CLASS(PSServer, service_param.server_class());
    TableManager::instance().initialize();
}

int32_t PSServer::configure(...){
    // for i in downpour_param.downpour_table_param_size()
    auto *table = CREATE_PSCORE_CLASS(
        Table, downpour_param.downpour_table_param(i).table_class());
    table->set_program_env(scope_.get(), place_, &server_sub_program);
    table->set_shard(_rank, shard_num);
    table->initialize(downpour_param.downpour_table_param(i),
                      config.fs_client_param());
    _table_map[downpour_param.downpour_table_param(i).table_id()].reset(table);

    return initialize();
}
```

### BrpcPsServer

```cpp
// paddle/fluid/distributed/ps/service/brpc_ps_server.h
class BrpcPsServer : public PSServer {
    brpc::Server _server;
}

// paddle/fluid/distributed/ps/service/brpc_ps_server.cc
int32_t BrpcPsServer::initialize() {
    auto *service =
      CREATE_PSCORE_CLASS(PsBaseService, service_config.service_class());
    _server.AddService(service, brpc::SERVER_DOESNT_OWN_SERVICE)
}
uint64_t BrpcPsServer::start(const std::string &ip, uint32_t port) {
    auto trainers = _environment->get_trainers(); // 可以去掉
    _server.Start(ip_port.c_str(), &options)
    _environment->registe_ps_server(ip, port, _rank);
}
```

### BrpcPsService

```cpp
class BrpcPsService : public PsBaseService {
  int32_t initialize_shard_info(...)
  int32_t pull_dense(...)
  int32_t push_dense(...)
  int32_t push_dense_param(...)
  int32_t push_sparse_param(...)
  int32_t pull_sparse(...)
  int32_t pull_geo_param(...)
  int32_t barrier(...)
  int32_t push_sparse(...)
  int32_t load_one_table(...)
  int32_t load_all_table(...)
  int32_t save_one_table(...)
  int32_t save_all_table(...)
  int32_t shrink_table(...)
  int32_t clear_one_table(...)
  int32_t clear_all_table(...)
  int32_t stop_server(...)
  int32_t start_profiler(...)
  int32_t stop_profiler(...)
  int32_t print_table_stat(...)
  int32_t push_global_step(...)
}
```

## fleet run_worker

### runtime

```python
# runtime 层初始化
class TheOnePSRuntime(RuntimeBase):
    def _init_worker(self, scopes=None):
        # in init
        # self._worker = fluid.core.DistFleetWrapper()
        self._worker.init_worker(proto_txt, self.string_hosts, role_id)
        # GEO mode
        self._communicator = Communicator(...)
        self._communicator.init_with_ctx(...)
        # 
        info = self._worker.get_client_info()
        self._worker.set_clients(all_info) # _all_gather info is all_info
        self._worker.create_client2client_connection()
        #
        self._pull_all_dense(scopes, send_ctx, dense_map)
        # GEO mode
        self._communicator.start()    

    def _pull_all_dense(self, scopes, send_ctx, recv_map):
        for name, ctx in send_ctx.items():
            self._worker.pull_dense_params(scope, table_id, var_names)
```

### init worker

### FleetWrapper

```cpp
// paddle/fluid/distributed/ps/wrapper/fleet.cc
void FleetWrapper::InitWorker(...){
    ps_env_.set_ps_servers(&host_sign_list, servers);
    worker_ptr_ = std::shared_ptr<paddle::distributed::PSClient>(
          paddle::distributed::PSClientFactory::create(ps_param));
    worker_ptr_->configure(ps_param, dense_pull_regions, ps_env_, index);
}

void FleetWrapper::PullDenseVarsSync(...){
    auto status = worker_ptr_->pull_dense(regions.data(), regions.size(), tid);
    status.wait();
}

int FleetWrapper::SetClients(std::vector<uint64_t>& host_sign_list) {
    return ps_env_.set_ps_clients(host_sign_list.data(), node);
}
void FleetWrapper::CreateClient2ClientConnection() {
    worker_ptr_->create_client2client_connection(...)
}
```

### PSClient

```cpp
// paddle/fluid/distributed/ps/service/ps_client.cc

PSClient *PSClientFactory::create(const PSParameter &ps_config) {
    PSClient *client = CREATE_PSCORE_CLASS(PSClient, service_param.client_class());
    TableManager::instance().initialize();
}

int32_t PSClient::configure(...){
    // for i in work_param.downpour_table_param_size()
    auto *accessor = CREATE_PSCORE_CLASS(
        ValueAccessor,
        work_param.downpour_table_param(i).accessor().accessor_class());
    accessor->configure(work_param.downpour_table_param(i).accessor());
    accessor->initialize();
    _table_accessors[work_param.downpour_table_param(i).table_id()].reset(accessor);
    return initialize();
}
```

### BrpcPsClient

```cpp
// paddle/fluid/distributed/ps/service/brpc_ps_client.cc
class BrpcPsClient : public PSClient {
    brpc::Server _server;
    DownpourPsClientService _service;
}

int32_t BrpcPsClient::initialize() {
    // for i in server_list.size()
    _server_channels[i][j].reset(new brpc::Channel());
    _server_channels[i][j]->Init(server_ip_port.c_str(), "", &options)
    // 启动client探听接口, 并相互建立连接
    start_client_service();
    // 异步push 请求队列初始化
    _push_dense_task_queue_map[table_id] = paddle::framework::MakeChannel<DenseAsyncTask *>();
    _push_sparse_task_queue_map[table_id] = paddle::framework::MakeChannel<SparseAsyncTask *>();
    // 启动异步push线程
    _async_push_sparse_thread = std::thread(std::bind(&BrpcPsClient::push_sparse_task_consume, this));
    // _async_push_sparse_thread.detach();
    _async_push_dense_thread = std::thread(std::bind(&BrpcPsClient::push_dense_task_consume, this));
}

// 启动client端RpcService 用于数据互发等操作
int32_t BrpcPsClient::start_client_service() {
    _service.configure(this, _client_id)
    _server.AddService(&_service, brpc::SERVER_DOESNT_OWN_SERVICE);
    _server.Start(butil::my_ip_cstr(), brpc::PortRange(start_port, max_port), &options)
    _env->registe_ps_client(...)
}

// how 弹性？？？
int32_t BrpcPsClient::create_client2client_connection(...){
    // for i in client_list.size()
    _client_channels[i].reset(new brpc::Channel());
    _client_channels[i]->Init(server_ip_port.c_str(), "", &options)
}
```

### DownpourPsClientService

```cpp
// paddle/fluid/distributed/ps/service/brpc_ps_client.cc

class DownpourPsClientService : public PsService {
    PSClient *_client;
    void service(...)
}
```

## communicator

```python
# python/paddle/fluid/communicator.py

class Communicator(object):
    def init_with_ctx(self,...):
        self.communicator_ = core.DistCommunicator(self.mode,...)
    def start(self):
        # Start communicator. Should call before training process.
        self.communicator_.start()
```

### bind

```cpp
// paddle/fluid/pybind/communicator_py.cc
void BindCommunicator(py::module* m) {
  // Communicator is already used by nccl, change to DistCommunicator
  py::class_<Communicator, std::shared_ptr<Communicator>>(*m, "DistCommunicator")
  .def(py::init([](...){Communicator::InitInstance<GeoCommunicator>(...)}
  .def("start", &Communicator::Start)
// paddle/fluid/distributed/ps/service/communicator/communicator.h
static Communicator *InitInstance(...){
    std::call_once(init_flag_, &Communicator::InitWithRpcCtx<T>,...);
}
static void InitWithRpcCtx(...){
    communicator_.reset(new T(std::ref(envs)));
    communicator_->InitEnvs();
    communicator_->InitBrpcClient(dist_desc, host_sign_list);
    communicator_->InitImpl(send_ctx, recv_ctx, recv_scope);
}
```

```cpp
// paddle/fluid/distributed/ps/service/communicator/communicator.cc
void Communicator::InitBrpcClient(...){
    auto fleet = paddle::distributed::FleetWrapper::GetInstance();
    _worker_ptr = fleet->worker_ptr_;
}
void AsyncCommunicator::InitImpl(...){
    // for varnames
    send_varname_to_queue_[var_name] = std::make_shared<BlockingQueue<std::shared_ptr<Variable>>>(send_queue_size_);
    send_threadpool_.reset(new ::ThreadPool(thread_pool_size_));
    }

void AsyncCommunicator::Start() {
    main_thread_.reset(new std::thread(std::bind(&AsyncCommunicator::MainThread, this))); // MainThread/RecvThread
}

void AsyncCommunicator::MainThread() {
    while (running_) {
        SendByCommunicator();
        RpcProfilerControl();
    }
}
void AsyncCommunicator::RecvThread() {
    while (running_) {
        RecvByCommunicator();
    }
}
```

## train_from_dataset

```python
# dataset
dataset = paddle.distributed.InMemoryDataset() # "MultiSlotInMemoryDataFeed"
dataset.load_into_memory()
dataset.init(...)
dataset.set_filelist(train_files_list)

# InMemoryDataset -- MultiSlotInMemoryDataFeed  -- InMemoryDataFeed -- DataFeed
# QueueDataset -- MultiSlotDataFeed -- PrivateQueueDataFeed -- DataFeed
# python/paddle/fluid/executor.py
```

```python
# class Executor(object):
def train_from_dataset(self,...):
    return self._run_from_dataset(...)

def _run_from_dataset(self,...):
    # dataset
    dataset = paddle.fluid.DatasetFactory().create_dataset(...)
    dataset.set_xxx(...)
    dataset._prepare_to_run()
    # trainer
    scope, trainer = self._prepare_trainer(...)
    trainer._gen_trainer_desc()
    # self._default_executor = core.Executor(p)
    trainer_instance = self._default_executor.init_for_dataset(
                    program.desc, trainer._desc(), scope, dataset.dataset)
    # run
    self._default_executor.run_from_dataset(trainer_instance)

def _prepare_trainer(self,...):
    trainer = TrainerFactory()._create_trainer(program.program._fleet_opt)
    # trainer._set_thread(thread)
```

### excutor

```cpp
// paddle/fluid/framework/executor.cc

std::shared_ptr<TrainerBase> Executor::InitForDataset(...){
  // MultiTrainer
  std::shared_ptr<TrainerBase> trainer;
  trainer = TrainerFactory::CreateTrainer(trainer_desc.class_name());
  // initialize trainer
  trainer->Initialize(trainer_desc, dataset);
  trainer->SetScope(scope);
  // prepare training environment and helper environment
  trainer->InitTrainerEnv(main_program, place_);
  // Try to init other environment
  trainer->InitOtherEnv(main_program);
}

void Executor::RunFromDataset(std::shared_ptr<TrainerBase> trainer) {
    trainer->Run();
}
```

### MultiTrainer

```cpp
//paddle/fluid/framework/trainer.h
class MultiTrainer : public TrainerBase {
    std::vector<DataFeed*> readers_;
    std::vector<std::shared_ptr<DeviceWorker>> workers_;
}

// paddle/fluid/framework/multi_trainer.cc
void MultiTrainer::Initialize(const TrainerDesc& trainer_desc, Dataset* dataset) {
    // Dataset -> DataFeed
    const std::vector<paddle::framework::DataFeed*> readers = dataset->GetReaders();
    thread_num_ = readers.size(); // !!! thread num
    workers_.resize(thread_num_); 
    // for i in thread_num_
    workers_[i] = DeviceWorkerFactory::CreateDeviceWorker(...)
    workers_[i]->Setxxx()
    workers_[i]->Initialize(trainer_desc);
    workers_[i]->SetDataFeed(readers[i]);
}

void MultiTrainer::Run() {
    // for i in thread_num_
    threads_.push_back(std::thread(&DeviceWorker::TrainFiles, workers_[thidx].get()));
    // for th in threads_
    th.join();
}
```

### HogwildWorker

```cpp
// paddle/fluid/framework/device_worker.cc
void DeviceWorker::SetDataFeed(DataFeed* data_feed) {
  device_reader_ = data_feed;
}

// paddle/fluid/framework/hogwild_worker.cc
void HogwildWorker::Initialize(const TrainerDesc &desc) {
}

void HogwildWorker::TrainFiles() {
    device_reader_->Start();
    while ((cur_batch = device_reader_->Next()) > 0) {
        // for op in ops_
        op->Run(*thread_scope_, place_);
    }
}
```

### MultiSlotInMemoryDataFeed

```cpp
// paddle/fluid/framework/data_feed.cc

class InMemoryDataFeed : public DataFeed {
    // 下面的 channel 赋值在 DatasetImpl<T>::CreateReaders()
    // input 为全局，output 和 consume 独立
    paddle::framework::ChannelObject<T>* input_channel_;
    paddle::framework::ChannelObject<T>* output_channel_;
    paddle::framework::ChannelObject<T>* consume_channel_;
}

bool InMemoryDataFeed<T>::Start() {
    //  input
    channel
    global channel
    input_channel_->Read(data); 
    output_channel_->Write(std::move(data));
}

int InMemoryDataFeed<T>::Next() {
    while (index < this->default_batch_size_) {
        output_channel_->Get(instance);
        ins_vec.push_back(instance);
        ++index;
        consume_channel_->Put(std::move(instance));
    }
    PutToFeedVec(ins_vec);
}

class MultiSlotInMemoryDataFeed : public InMemoryDataFeed<Record> {
}
```

### MultiSlotDataFeed

```cpp
// paddle/fluid/framework/data_feed.cc

class PrivateQueueDataFeed : public DataFeed {
    std::shared_ptr<paddle::framework::ChannelObject<T>> queue_;
}

bool PrivateQueueDataFeed<T>::Start() {
    read_thread_ = std::thread(&PrivateQueueDataFeed::ReadThread, this);
}
void PrivateQueueDataFeed<T>::ReadThread() {
    while (PickOneFile(&filename)) {
        fp_ = fs_open_read(filename, &err_no, pipe_command_);
        while (ParseOneInstanceFromPipe(&instance)) {
            queue_->Put(instance);
        }
    }
}
int PrivateQueueDataFeed<T>::Next() {
    while (index < default_batch_size_) {
        queue_->Get(instance)
        AddInstanceToInsVec(&ins_vec, instance, index++);
    }
    PutToFeedVec(ins_vec);
}

class MultiSlotDataFeed : public PrivateQueueDataFeed<std::vector<MultiSlotType>> {
}
```

#### Misc

1. InMemoryDataset 流程分析
* LoadIntoMemory 把文件读取进 input_channel_，注意 input_channel_ 是全局共享，由 GetReaders() 返回时设定；

* Start() 从 input_channel_ 读取一份数据进 output_channel_

* Next() 从 output_channel_ 取数据进 consume_channel_
2. input_channel_ 在哪里初始化？
   data_set.cc 中 DatasetImpl<T>::CreateChannel()，它是全局的，最终调用在 dataset.py 中 self.dataset.create_channel()，所以 InMemoryDataset 有调用，QueueDataset 没有调用

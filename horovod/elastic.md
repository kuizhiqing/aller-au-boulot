# Elastic


## 弹性启动

前述接正常启动部分，弹性使用 gloo 启动

```python
def _run_elastic(args):
    settings = elastic_settings.ElasticSettings(discovery=discover_hosts,...)
    return gloo_run_elastic(settings, env, args.run_func if args.run_func else args.command, executable)

from horovod.runner.gloo_run import gloo_run, gloo_run_elastic
```

```python
# horovod/runner/gloo_run.py

def gloo_run_elastic(settings, env, command_or_func, executable) -> Optional[Any]:
    rendezvous = RendezvousServer(settings.verbose)
    return launch_gloo_elastic(command_or_func, exec_command, settings, env, get_common_interfaces, rendezvous, executable)

def launch_gloo_elastic(command_or_func, exec_command, settings, env, get_common_interfaces, rendezvous, executable):
    driver = ElasticDriver(rendezvous, settings.discovery,
                           settings.min_num_proc, settings.max_num_proc,
                           timeout=settings.elastic_timeout,
                           reset_limit=settings.reset_limit,
                           cooldown_range=settings.cooldown_range,
                           verbose=settings.verbose)

    handler = create_rendezvous_handler(driver)
    global_rendezv_port = rendezvous.start(handler)
    driver.wait_for_available_slots(settings.num_proc)

    run_command = get_run_command(command, server_ip, nics, global_rendezv_port, elastic=True)
    create_worker = _create_elastic_worker_fn(exec_command, run_command, env, event)

    driver.start(settings.num_proc, create_worker)
    res = driver.get_results()
    driver.stop()

```

### Driver

ElasticDriver 中有两个线程

* 一个线程 while 循环执行 `_discover_hosts` 监控、报告节点更新
* 一个线程执行 `run_worker` 启动 worker

注意 driver 不会阻塞，启动 worker 后返回 launch 函数，等待获取 results, 具体地，`driver.get_results()` 是一个 while 循环会造成阻塞。

```python
# horovod/runner/elastic/driver.py 

class ElasticDriver(object):
    def __init__(self, rendezvous, ...):
        self._worker_registry = WorkerStateRegistry(self, self._host_manager, reset_limit=reset_limit)
        self._results = ResultsRecorder()
        self._discovery_thread = threading.Thread(target=self._discover_hosts)

    def start(self, num_proc, create_worker_fn):
        self._activate_workers(num_proc)

    def get_results(self):
        return self._results.get_results()

    def register_worker_server(self, host, slot, addresses, secret_key):
        self._worker_clients[(host, slot)] = WorkerNotificationClient(
            addresses, secret_key, self._verbose)

    def _activate_workers(self, min_num_proc):
        current_hosts = self.wait_for_available_slots(min_num_proc)
        pending_slots = self._update_host_assignments(current_hosts)
        self._worker_registry.reset(self.world_size())
        self._start_worker_processes(pending_slots)

    def _discover_hosts(self):
        while not self._shutdown.is_set():
            self._wait_hosts_cond.acquire()
            try:
                update_res = self._host_manager.update_available_hosts()
                if update_res != HostUpdateResult.no_update:
                    self._notify_workers_host_changes(self._host_manager.current_hosts, update_res)
                    self._wait_hosts_cond.notify_all()
            finally:
                self._wait_hosts_cond.release()
            self._shutdown.wait(DISCOVER_HOSTS_FREQUENCY_SECS)

    def _notify_workers_host_changes(self, current_hosts, update_res):
        coordinator_client.notify_hosts_updated(timestamp, update_res)

    def _start_worker_processes(self, pending_slots):
        for slot_info in pending_slots:
            self._start_worker_process(slot_info)

    def _start_worker_process(self, slot_info):
        def run_worker():
            res = create_worker_fn(slot_info, [shutdown_event, host_event])
            exit_code, timestamp = res
            self._handle_worker_exit(slot_info, exit_code, timestamp)

        thread = threading.Thread(target=run_worker)
        thread.daemon = True
        thread.start()
        self._results.expect(thread)

```

## Usage

```python
hvd.init()

torch.cuda.set_device(hvd.local_rank())

dataset = ...
model = ...

optimizer = optim.SGD(model.parameters(), lr * hvd.size())
optimizer = hvd.DistributedOptimizer(optimizer)

@hvd.elastic.run
def train(state):
    batch_offset = state.batch
    for state.epoch in range(state.epoch, epochs):
        for state.batch in range(state.batch, batches_per_epoch):
            data, target = get_random_batch()

            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

            if state.batch % batches_per_commit == 0:
                state.commit()
        state.batch = 0

def on_state_reset():
    # adjust learning rate on reset
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * hvd.size()

state = hvd.elastic.TorchState(model, optimizer, batch=0, epoch=0)
state.register_reset_callbacks([on_state_reset])
train(state)
```

## Implementation

在训练的循环函数上会有装饰器，用于识别错误和自身的恢复，另外通过 `notification_manager` 别的节点也会感知到错误的发生。

```python
# horovod/common/elastic.py

notification_manager = WorkerNotificationManager()

def run_fn(func, reset):
    @functools.wraps(func)
    def wrapper(state, *args, **kwargs):
        notification_manager.init()
        notification_manager.register_listener(state)

        try:
            while True:
                try:
                    if not skip_sync:
                        state.sync()

                    return func(state, *args, **kwargs)
                except HorovodInternalError:
                    state.restore()
                    skip_sync = False
                except HostsUpdatedInterrupt as e:
                    skip_sync = e.skip_sync

                reset()
                state.on_reset()
    return wrapper
```


```python
# horovod/common/elastic.py

class State(object):
    def register_reset_callbacks(self, callbacks):
        pass

    def on_reset(self):
        self._host_messages = queue.Queue()
        self.reset()
        for callback in self._reset_callbacks:
            callback()

    def on_hosts_updated(self, timestamp, update_res):
        self._host_messages.put((timestamp, update_res))

    def commit(self):
        self.save()
        self.check_host_updates()

    def check_host_updates(self):
        raise HostsUpdatedInterrupt(all_update == HostUpdateResult.removed)

    def save(self):
        pass

    def restore(self):
        pass

    def sync(self):
        pass

    def reset(self):
        pass

class ObjectState(State):
    def restore(self):
        self._set_attrs()

    def sync(self):
        if self._saved_state:
            self._saved_state = self._bcast_object(self._saved_state)
            self._set_attrs()


# horovod/torch/elastic/state.py

class TorchState(ObjectState):
    def restore(self):
        for handler in self._handlers.values():
            handler.restore()
        super(TorchState, self).restore()
```

因为 `hvd.init()` 之后，一直会有后台进程在负责真正的通信过程，在有节点变化时，通信组需要重建，具体实现如下。

```cpp
// horovod/common/operations.cc

bool RunLoopOnce(HorovodGlobalState& state) {
  if (state.dynamic_process_sets) {
    // Initialize any newly added process set that has been registered by all
    // Horovod processes and remove a process set that has been marked for
    // removal by all Horovod processes.
    if (state.control_operation == LibType::GLOO) {
      state.process_set_table.InitializeRegisteredAndRemoveMarkedIfReady(
          global_gloo_context);
    }
  }

}
```

Reinitialize Horovod context performing a new round of rendezvous.

负责重建通信域

```cpp
// horovod/common/process_set.cc

template <class Context>
int32_t ProcessSetTable::InitializeRegisteredAndRemoveMarkedIfReady_(
    const Context& global_context, const Status& removal_status) {

  if (registered_count_agreement) {
    for (auto id : Ids()) {
      bool newly_registered = Get(id).Initialize(global_context);
    }
  }

  if (id_to_be_removed_agreement) {
      id_to_process_set_[id_to_be_removed_].Finalize(removal_status);
      DeregisterProcessSet(id_to_be_removed_);
  }
}

ProcessSetTable::ProcessSetTable() {
  auto process_set_id = RegisterProcessSet();
}

int32_t ProcessSetTable::RegisterProcessSet(std::vector<int> global_ranks) {
  // ProcessSet
  ids_.push_back(id);
}

bool ProcessSet::Initialize(const GlooContext& global_gloo_context) {
  gloo_context.InitializeForProcessSet(global_gloo_context,
                                       registered_global_ranks);
}
```

```cpp
// horovod/common/gloo/gloo_context.cc

void GlooContext::InitializeForProcessSet(const GlooContext& global_context,
                                          const std::vector<int>& registered_ranks) {
  ctx = Rendezvous(HOROVOD_GLOO_GLOBAL_PREFIX + process_set_suffix,
                   rendezvous_addr_env, rendezvous_port, rank, size, dev,
                   timeout_);
}
```

接口的调用顺序 

```python
# horovod/common/process_sets.py

def add_process_set(process_set: Union[ProcessSet, Sequence[int]]) -> ProcessSet:
    process_set_id = _basics._add_process_set_impl(process_set.ranks)
```

```python
# horovod/common/basics.py

class HorovodBasics(object):
    def _add_process_set_impl(self, ranks: Sequence[int]) -> Optional[int]:
        result = int(self.MPI_LIB_CTYPES.horovod_add_process_set(
            (ctypes.c_int * nrank)(*ranks), ctypes.c_int(nrank)))
```


```cpp
// horovod/common/operations.cc

int horovod_add_process_set(const int* ranks, int nrank) {
  ProcessSet* process_set = nullptr;
  {
    process_set = &GetProcessSetOrAddUnitialized(
        ranks && nrank > 0 ? std::vector<int>(ranks, ranks + nrank) : std::vector<int>(),
        id);
  }
}

ProcessSet& GetProcessSetOrAddUnitialized(std::vector<int> ranks, int& id) {
  id = horovod_global.process_set_table.FindId(ranks);
  id = horovod_global.process_set_table.RegisterProcessSet(std::move(ranks));
  auto& process_set = horovod_global.process_set_table.Get(id);
  EnrichProcessSetWithGlooController(process_set);
}

void EnrichProcessSetWithGlooController(ProcessSet& process_set) {
  process_set.controller.reset(new GlooController(
      process_set.response_cache, process_set.tensor_queue,
      horovod_global.timeline, horovod_global.parameter_manager,
      process_set.group_table, horovod_global.timeline_controller,
      process_set.gloo_context));
}
```

### Workflow Summary

* `hvd.elastic.run` decorator 
* process `HorovodInternalError` 
* reinitialize context, new rendezvous
* restore state, broadcasting from new worker-0

### Reference

* https://horovod.readthedocs.io/en/stable/elastic_include.html

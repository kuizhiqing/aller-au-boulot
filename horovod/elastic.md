# Elastic


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

```python
# horovod/common/elastic.py

def run_fn(func, reset):
    @functools.wraps(func)
    def wrapper(state, *args, **kwargs):
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

调用顺序 

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

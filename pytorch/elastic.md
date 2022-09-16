# Ealstic

## launch/run

```shell
python -m torch.distributed.run
```

模块实际调用 `elastic_launch` 函数启动

```python
# torch/distributed/run.py

def run(args):
    config, cmd, cmd_args = config_from_args(args)
    elastic_launch(
        config=config,
        entrypoint=cmd,
    )(*cmd_args)


@record
def main(args=None):
    args = parse_args(args)
    run(args)

if __name__ == "__main__":
    main()
```

elastic_launch 调用 launch_agent 方法

* 创建 RendezvousParameters，只包含声明
* 创建 WorkerSpec, rdzv_handler 参数处理见 rendezvous 部分
* 创建 LocalElasticAgent
* 调用 agent.run()

```python
# torch/distributed/launcher/api.py

class elastic_launch:
    def __call__(self, *args):
        return launch_agent(self._config, self._entrypoint, list(args))

def launch_agent(...)
    run_id = str(uuid.uuid4().int)

    entrypoint_name = _get_entrypoint_name(entrypoint, args)

    rdzv_parameters = RendezvousParameters(...)

    # 这里的 master 只有在 rdzv_backend == static 时等于 rdzv, 否则都是 None，将会在后面创建
    master_addr, master_port = _get_addr_and_port(rdzv_parameters)

    spec = WorkerSpec(
        role=config.role,
        local_world_size=config.nproc_per_node,
        entrypoint=entrypoint,
        args=tuple(args),
        rdzv_handler=rdzv_registry.get_rendezvous_handler(rdzv_parameters),
        max_restarts=config.max_restarts,
        monitor_interval=config.monitor_interval,
        redirects=config.redirects,
        tee=config.tee,
        master_addr=master_addr,
        master_port=master_port,
    )

    agent = LocalElasticAgent(
        spec=spec, start_method=config.start_method, log_dir=config.log_dir
    )

    result = agent.run()
```

## rendezvous

rendezvous 模块在初始化时把默认支持的 handler 都进行了初始化，注册在 handler_registry 中。

```python
# torch/distributed/elastic/rendezvous/__init__.py
from .registry import _register_default_handlers
_register_default_handlers()
```

即提供了对应关系，可以通过 handler key 获取到 create handler 方法。

```python
# torch/distributed/elastic/rendezvous/registry.py

from .api import rendezvous_handler_registry as handler_registry
from .dynamic_rendezvous import create_handler

def _create_c10d_handler(params: RendezvousParameters) -> RendezvousHandler:
    from .c10d_rendezvous_backend import create_backend

    backend, store = create_backend(params)
    return create_handler(store, backend, params)

def _register_default_handlers() -> None:
    handler_registry.register("c10d", _create_c10d_handler)
    handler_registry.register("static", _create_static_handler)


def get_rendezvous_handler(params: RendezvousParameters) -> RendezvousHandler:
    return handler_registry.create_handler(params)
```

> 注意这里有两个 create_handler，一个从注册器中取出并调用，一个是 create backend 后的封装。

启动时调用的 `rdzv_registry.get_rendezvous_handler(rdzv_parameters)` 即通过 prameter 获取对应 handler 并初始化。

```python
import torch.distributed.elastic.rendezvous.registry as rdzv_registry
```

```python
#  torch/distributed/elastic/rendezvous/api.py

rendezvous_handler_registry = RendezvousHandlerRegistry()

class RendezvousHandlerRegistry:

    _registry: Dict[str, RendezvousHandlerCreator]

    def register(self, backend: str, creator: RendezvousHandlerCreator) -> None:
        self._registry[backend] = creator

    def create_handler(self, params: RendezvousParameters) -> RendezvousHandler:
        creator = self._registry[params.backend]
        handler = creator(params)
        return handler
```


以 c10d 为例说明 create_backend，即真正启动服务的部分

```python
# torch/distributed/elastic/rendezvous/c10d_rendezvous_backend.py

def _create_tcp_store(params: RendezvousParameters) -> TCPStore:
    host, port = parse_rendezvous_endpoint(params.endpoint, default_port=29400)
    # 对于同一台机器多进程启动的 case，通过重试解决
    store = TCPStore(host, port, is_master=is_server, timeout=timedelta(seconds=read_timeout))

    return store

def create_backend(params: RendezvousParameters) -> Tuple[C10dRendezvousBackend, Store]:
    store_type = params.get("store_type", "tcp").strip().lower()
    store = _create_tcp_store(params)
    backend = C10dRendezvousBackend(store, params.run_id)

    return backend, store
```

```python
# torch/distributed/elastic/rendezvous/dynamic_rendezvous.py

def create_handler(
    store: Store, backend: RendezvousBackend, params: RendezvousParameters
) -> DynamicRendezvousHandler:
        return DynamicRendezvousHandler.from_backend(...)

class DynamicRendezvousHandler(RendezvousHandler):
    @classmethod
    def from_backend(
        cls,
        run_id: str,
        store: Store,
        backend: RendezvousBackend,
        min_nodes: int,
        max_nodes: int,
        timeout: Optional[RendezvousTimeout] = None,
    ):
        return cls(node, settings, backend.name, store, state_holder)

    def next_rendezvous(self) -> Tuple[Store, int, int]:
        self._start_heartbeats()

        rank, world_size = self._get_world()
        store = self._get_store()

        return store, rank, world_size

    def _keep_alive(self) -> None:
        ...

    def _start_heartbeats(self) -> None:
        ...

```

`_keep_alive`  是通过 `_PeriodicTimer` 启动线程依赖 backend 实现的。

链路逻辑，

* `_keep_alive_weak` 调用 `_keep_alive`，`_DistributedRendezvousOpExecutor.run` 方法声明更新
* `_DistributedRendezvousOpExecutor` 初始化时需要传入 `_state_holder`
* `state_holder: _RendezvousStateHolder` 在 DynamicRendezvousHandler` 初始化时传入
* _BackendRendezvousStateHolder(backend, settings) 使用 `backend`

## worker

**TL;DR;**

提供 WorkerSpec/Worker/WorkerGroup/WorkerState/RunResult 抽象, 封装 process 管理。

调用如前所述，初始化后使用 `agent.run()` 调用，

* run 调用 invoke_run
* invoke_run 调用 _initialize_workers 实际拉起 worker 进程，然后 while 循环监控状态

_initialize_workers

* 首先调用 _rendezvous: 0 号节点在 store 中写入 master 地址，所有节点从中取出 master 地址，并不使用，只为了做同步
* 调用 _start_workers 启动 worker

> 0 号节点写入的 master 地址在使用 c10d backend （非 static）时并不是 rendevous endpoint，默认情况会通过 socket bind 获取可用端口，然后写入 store，其余节点从 store 中获取。

```python
# torch/distributed/elastic/agent/server/api.py 

class SimpleElasticAgent(ElasticAgent):

    def run(self, role: str = DEFAULT_ROLE) -> RunResult:
        result = self._invoke_run(role)
        return result

    def _invoke_run(self, role: str = DEFAULT_ROLE) -> RunResult:
        self._initialize_workers(self._worker_group)
        rdzv_handler = spec.rdzv_handler

        while True:
            run_result = self._monitor_workers(self._worker_group)
            state = run_result.state
            if state == WorkerState.SUCCEEDED:
                self._exit_barrier()
                return run_result
            elif state in {WorkerState.UNHEALTHY, WorkerState.FAILED}:
                if self._remaining_restarts > 0:
                    self._remaining_restarts -= 1
                    self._restart_workers(self._worker_group)
                else:
                    self._stop_workers(self._worker_group)
                    self._worker_group.state = WorkerState.FAILED
                    self._exit_barrier()
                    return run_result
            elif state == WorkerState.HEALTHY:
                num_nodes_waiting = rdzv_handler.num_nodes_waiting()
                if num_nodes_waiting > 0:
                    self._restart_workers(self._worker_group)

    def _initialize_workers(self, worker_group: WorkerGroup) -> None:
        self._rendezvous(worker_group)
        worker_ids = self._start_workers(worker_group)

    def _rendezvous(self, worker_group: WorkerGroup) -> None:
        store, group_rank, group_world_size = spec.rdzv_handler.next_rendezvous()
        workers = self._assign_worker_ranks(store, group_rank, group_world_size, spec)

        if group_rank == 0:
            self._set_master_addr_port(store, spec.master_addr, spec.master_port)

        # 获取 master 地址，起到同步作用
        master_addr, master_port = self._get_master_addr_port(store)

    @staticmethod
    def _set_master_addr_port(
        store: Store, master_addr: Optional[str], master_port: Optional[int]
    ):
        if master_port is None:
            sock = _get_socket_with_port()
            with closing(sock):
                master_port = sock.getsockname()[1]

        if master_addr is None:
            master_addr = _get_fq_hostname()

        store.set("MASTER_ADDR", master_addr.encode(encoding="UTF-8"))
        store.set("MASTER_PORT", str(master_port).encode(encoding="UTF-8"))

    @staticmethod
    def _get_master_addr_port(store: Store) -> Tuple[str, int]:
        master_addr = store.get("MASTER_ADDR").decode(encoding="UTF-8")
        master_port = int(store.get("MASTER_PORT").decode(encoding="UTF-8"))
        return (master_addr, master_port)


def _get_socket_with_port() -> socket.socket:
    s = socket.socket(family, type, proto)
    s.bind(("localhost", 0))
    s.listen(0)
    return s

```

_start_workers 为真正启动进程的模块，

* 首先获取 master 地址，这个地址在非 static backend 时是 0 号节点写入 store 的
* 为 process 准备多种配置，主要包括 env，args，然后调用进程封装模块启动进程返回进程 id

```python
# torch/distributed/elastic/agent/server/local_elastic_agent.py

class LocalElasticAgent(SimpleElasticAgent):

    def _start_workers(self, worker_group: WorkerGroup) -> Dict[int, Any]:
        master_addr, master_port = super()._get_master_addr_port(store)
        for worker in worker_group.workers:
            local_rank = worker.local_rank
            worker_env = {
                "LOCAL_RANK": str(local_rank),
                "RANK": str(worker.global_rank),
                "GROUP_RANK": str(worker_group.group_rank),
                "ROLE_RANK": str(worker.role_rank),
                "ROLE_NAME": spec.role,
                "LOCAL_WORLD_SIZE": str(spec.local_world_size),
                "WORLD_SIZE": str(worker.world_size),
                "GROUP_WORLD_SIZE": str(worker_group.group_world_size),
                "ROLE_WORLD_SIZE": str(worker.role_world_size),
                "MASTER_ADDR": master_addr,
                "MASTER_PORT": str(master_port),
                "TORCHELASTIC_RESTART_COUNT": str(restart_count),
                "TORCHELASTIC_MAX_RESTARTS": str(spec.max_restarts),
                "TORCHELASTIC_RUN_ID": spec.rdzv_handler.get_run_id(),
                "TORCHELASTIC_USE_AGENT_STORE": str(use_agent_store),
                "NCCL_ASYNC_ERROR_HANDLING": os.getenv(
                    "NCCL_ASYNC_ERROR_HANDLING", str(1)
                ),
            }
            if "OMP_NUM_THREADS" in os.environ:
                worker_env["OMP_NUM_THREADS"] = os.environ["OMP_NUM_THREADS"]

            envs[local_rank] = worker_env
            worker_args = list(spec.args)
            worker_args = macros.substitute(worker_args, str(local_rank))
            args[local_rank] = tuple(worker_args)

        self._pcontext = start_processes(
            name=spec.role,
            entrypoint=spec.entrypoint,
            args=args,
            envs=envs,
            log_dir=attempt_log_dir,
            start_method=self._start_method,
            redirects=spec.redirects,
            tee=spec.tee,
        )

        return self._pcontext.pids()

```

子进程启动的用户脚本例如 trainer.py，会获取这里配置的环境运行。

## process

对进程和线程的封装，主要是 process 和 multiprocessing 库的封装。

通过对 entrypoint 是否是 str 判断决定启动方式。

```python
# torch/distributed/elastic/multiprocessing/__init__.py

def start_processes(
    name: str,
    entrypoint: Union[Callable, str],
) -> PContext:
    context: PContext
    if isinstance(entrypoint, str):
        context = SubprocessContext(...)
    else:
        context = MultiprocessContext(...)

    try:
        context.start()
        return context
    except Exception:
        context.close()
        raise
```

```python
# torch/distributed/elastic/multiprocessing/api.py

class SubprocessHandler:
    def __init__(...):
        self.proc: subprocess.Popen = self._popen(args_str, env_vars)

class SubprocessContext(PContext):
    def __init__(...):
        self._running_local_ranks: Set[int] = set(range(self.nprocs))
        self._failures: Dict[int, ProcessFailure] = {}
        self.subprocess_handlers: Dict[int, SubprocessHandler] = {}

    def _start(self):
        self.subprocess_handlers = { local_rank: SubprocessHandler(...) }

    def _poll(self) -> Optional[RunProcsResult]:
        for local_rank in self._running_local_ranks:
            exitcode = handler.proc.poll()
            if exitcode is not None:
                done_local_ranks.add(local_rank)
        self._running_local_ranks.difference_update(done_local_ranks)

    def _close(self, death_sig: signal.Signals, timeout: int = 30) -> None:
        for handler in self.subprocess_handlers.values():
            if handler.proc.poll() is None:
                handler.close(death_sig=death_sig)

class PContext(abc.ABC):
    def __init__(...):
        self.entrypoint = entrypoint

    def start(self) -> None:
        self._start()

    def wait(self, timeout: float = -1, period: float = 1) -> Optional[RunProcsResult]:
        expiry = time.time() + timeout
        while time.time() < expiry:
            pr = self._poll()
            if pr: return pr

    def close(
        self, death_sig: Optional[signal.Signals] = None, timeout: int = 30
    ) -> None:
        self._close(death_sig=death_sig, timeout=timeout)
```

```python
import torch.multiprocessing as mp

class MultiprocessContext(PContext):
    def __init__(..., entrypoint: Callable, ...):
        self._ret_vals = {
            local_rank: mp.get_context(self.start_method).SimpleQueue()
            for local_rank in range(self.nprocs)
        }
        self._pc: Optional[mp.ProcessContext] = None

    def _start(self):
        self._pc = mp.start_processes(...)

    def _poll(self) -> Optional[RunProcsResult]:
        self._pc.join(-1)

    def _close(self, death_sig: signal.Signals, timeout: int = 30) -> None:
        for proc in self._pc.processes:
            if proc.is_alive():
                try:
                    os.kill(proc.pid, death_sig)
```

## demo

```python
import torch

torch.distributed.init_process_group(backend="nccl", init_method="env://")
print(torch.distributed.get_world_size())
```

init_process_group 提供两种初始化方式

* 显式提供 store, rank, world_size 以初始化
* 指定 init_method, 默认为 env:// 使用环境变量，如 launch/run 模块已配置好了环境变量

在没有 store 的时候会通过 rendevous 创建 store 并获取 rank 和 size 信息。

然后根据这些信息创建 process group,

* mpi 从 orte 中获取信息，不需要通过这里指定
* gloo/nccl 通过 store, rank, size 初始化创建通信域

```python
# torch/distributed/distributed_c10d.py

from .rendezvous import rendezvous

def init_process_group(
    backend,
    init_method=None,
    timeout=default_pg_timeout,
    world_size=-1,
    rank=-1,
    store=None,
    group_name="",
    pg_options=None,
):
    backend = Backend(backend)

    if backend == Backend.MPI:
        default_pg = _new_process_group_helper(
            -1, -1, [], Backend.MPI, None, group_name=group_name, timeout=timeout
        )
        _update_default_pg(default_pg)
    else:
        if store is None:
            rendezvous_iterator = rendezvous(
                init_method, rank, world_size, timeout=timeout
            )
            store, rank, world_size = next(rendezvous_iterator)
            store = PrefixStore("default_pg", store)

        default_pg = _new_process_group_helper(...)
        _update_default_pg(default_pg)

    if backend == Backend.MPI:
        barrier()
    else:
        _store_based_barrier(rank, store, timeout)


def _new_process_group_helper(
    world_size,
    rank,
    group_ranks,
    backend,
    store,
    pg_options=None,
    group_name=None,
    timeout=default_pg_timeout,
):
    backend = Backend(backend)
    if backend == Backend.MPI:
        pg = ProcessGroupMPI.create(group_ranks)
        _pg_map[pg] = (Backend.MPI, None)
        _pg_names[pg] = group_name
    else:
        prefix_store = PrefixStore(group_name, store)

        if backend == Backend.GLOO:
            pg = ProcessGroupGloo(prefix_store, rank, world_size, timeout=timeout)
            _pg_map[pg] = (Backend.GLOO, store)
            _pg_names[pg] = group_name
        elif backend == Backend.NCCL:
            pg = ProcessGroupNCCL(prefix_store, rank, world_size, pg_options)
            _pg_map[pg] = (Backend.NCCL, store)
            _pg_names[pg] = group_name

    return pg

```

rendevous 会创建 store 并返回 rank, size 信息，创建 store 通过 `_create_c10d_store` 实现，
在调用c api 创建 TCPStore 时通过 start_daemon 指定是否在当前调用里创建服务。

```python
# torch/distributed/rendezvous.py

_rendezvous_handlers = {}

def rendezvous(url: str, rank: int = -1, world_size: int = -1, **kwargs):
    return _rendezvous_handlers[result.scheme](url, **kwargs)

def register_rendezvous_handler(scheme, handler):
    _rendezvous_handlers[scheme] = handler

register_rendezvous_handler("tcp", _tcp_rendezvous_handler)
register_rendezvous_handler("env", _env_rendezvous_handler)
register_rendezvous_handler("file", _file_rendezvous_handler)

def _file_rendezvous_handler(url: str, **kwargs):
    result = urlparse(url)
    query_dict = _query_to_dict(result.query)
    rank = int(query_dict["rank"])
    world_size = int(query_dict["world_size"])
    store = FileStore(path, world_size)
    yield (store, rank, world_size)


def _create_c10d_store(hostname, port, rank, world_size, timeout) -> Store:
    if _torchelastic_use_agent_store():
        tcp_store = TCPStore(hostname, port, world_size, False, timeout)
        return PrefixStore(f"/worker/attempt_{attempt}", tcp_store)
    else:
        start_daemon = rank == 0
        return TCPStore(
            hostname, port, world_size, start_daemon, timeout, multi_tenant=True
        )


def _tcp_rendezvous_handler(
    url: str, timeout: timedelta = default_pg_timeout, **kwargs
):
    result = urlparse(url)
    query_dict = _query_to_dict(result.query)
    rank = int(query_dict["rank"])
    world_size = int(query_dict["world_size"])

    store = _create_c10d_store(result.hostname, result.port, rank, world_size, timeout)

    yield (store, rank, world_size)


def _env_rendezvous_handler(
    url: str, timeout: timedelta = default_pg_timeout, **kwargs
):
    result = urlparse(url)
    query_dict: Dict[str, Union[int, str]] = _query_to_dict(result.query)

    if "rank" in query_dict:
        rank = int(query_dict["rank"])
    else:
        rank = int(_get_env_or_raise("RANK"))

    if "world_size" in query_dict:
        world_size = int(query_dict["world_size"])
    else:
        world_size = int(_get_env_or_raise("WORLD_SIZE"))

    master_addr = _get_env_or_raise("MASTER_ADDR")
    master_port = int(_get_env_or_raise("MASTER_PORT"))

    store = _create_c10d_store(master_addr, master_port, rank, world_size, timeout)

    yield (store, rank, world_size)

```

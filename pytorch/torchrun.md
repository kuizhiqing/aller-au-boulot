# torchrun

可以通过以下命令启动一个 2 机 8 卡的分布式训练任务，该命令需要在所有 2 个节点上执行。

```bash
torchrun 
    --nnodes=2 
    --nproc-per-node=8 
    --rdzv-endpoint=123.45.67.89:36123 
    --rdzv-backend=c10d
    demo.py
```

根据环境变量转换规则，[env action](https://github.com/pytorch/pytorch/blob/main/torch/distributed/argparse_util.py#L13-L58)， 上述启动命令等价于以下命令：

```bash
export PET_NPROC_PER_NODE=8
export PET_NNODES=2
export PET_RDZV_ENDPOINT=123.45.67.89:36123
export PET_RDZV_BACKEND=c10d

torchrun demo.py
```

以下 allreduce 的例子可以用于测试完整流程。

```python
# demo.py

import torch
torch.distributed.init_process_group(backend="nccl", init_method="env://")
rank = torch.distributed.get_rank()
torch.cuda.set_device(rank % torch.cuda.device_count())
world_size = torch.distributed.get_world_size()
print(f"rank {rank} world_size {world_size}")
a = torch.tensor([1]).cuda()
torch.distributed.all_reduce(a)
print(f"rank {rank} world_size {world_size} {a}")
torch.distributed.barrier()
print(f"rank {rank} world_size {world_size}")
```

下面分析启动的细节流程。

## run

根据 `setup.py` 可以看出 `torchrun` 对应的启动函数

```python
# setup.py

def configure_extension_build():
    entry_points = {
        "console_scripts": [
            "torchrun = torch.distributed.run:main",
        ],
    }
```

在 pytorch 1.9.0 版本后引入 `torch.distributed.run` 模块取代 `torch.distributed.launch` 启动分布式任务并支持弹性容错能力。

```python
# torch/distributed/run.py

from torch.distributed.launcher.api import elastic_launch

def run(args):
    config, cmd, cmd_args = config_from_args(args)
    elastic_launch(config=config, entrypoint=cmd,)(*cmd_args)

@record
def main(args=None):
    args = parse_args(args)
    run(args)

if __name__ == "__main__":
    main()
```

可以看到实际执行的是伪装成函数的 `elastic_launch` 类

```python
# torch/distributed/launcher/api.py

class elastic_launch:
    def __init__(self, config: LaunchConfig, entrypoint: Union[Callable, str, None],):
        self._config = config
        self._entrypoint = entrypoint

    def __call__(self, *args):
        return launch_agent(self._config, self._entrypoint, list(args))

def launch_agent(config: LaunchConfig, entrypoint: Union[Callable, str, None], args: list[Any],):
    rdzv_parameters = RendezvousParameters(
        backend=config.rdzv_backend,
        endpoint=config.rdzv_endpoint,
        run_id=config.run_id,
        min_nodes=config.min_nodes,
        max_nodes=config.max_nodes,
        local_addr=config.local_addr,
        **config.rdzv_configs,
    )

    master_addr, master_port = _get_addr_and_port(rdzv_parameters)

    spec = WorkerSpec(
        role=config.role,
        local_world_size=config.nproc_per_node,
        entrypoint=entrypoint,
        args=tuple(args),
        rdzv_handler=rdzv_registry.get_rendezvous_handler(rdzv_parameters),
        master_addr=master_addr,
        master_port=master_port,
    )

    agent = LocalElasticAgent(spec=spec, ...)

    try:
        result = agent.run()
        return result.return_values

def _get_addr_and_port(rdzv_parameters: RendezvousParameters,) -> tuple[Optional[str], Optional[int]]:
    if rdzv_parameters.backend != "static":
        return (None, None)

    endpoint = rdzv_parameters.endpoint
    endpoint = endpoint.strip()
    master_addr, master_port = parse_rendezvous_endpoint(endpoint, default_port=-1)
    return (master_addr, master_port)
```

`elastic_launch` 通过 `launch_agent` 实现了主要的启动流程。

* 通过启动参数定义 **rendezvous**, 用于节点间的协同模块
* 定义进程 **worker** 的描述信息 WorkerSpec，一个 worker 对应一个进程，一般对应一个 GPU
* 定义并启动 **agent**, LocalElasticAgent 在每个分布式节点上启动，管理节点上的多个 worker 进程

注意到：

* 当 rendezvous backend 为 `static` 时，worker 中的 `master_addr` 和 `master_port` 为 `None`， 否则比如为 c10d 时，`master_addr` 和 `master_port` 为 endpoint 中的 ip 和 port.
* 根据 `rendezvous backend` 参数会从 `rdzv_registry` 中选择对应的 `rendezvous handler`，比如 `etcd`，`c10d` 等，不同的 handler 采用不同的方式实现 rendezvous 即分布式节点间如何协同.

## worker

worker 并没有被封装成 process 的抽象，窃以为这里是有讨论空间的。

所以 WorkerSpec/Worker 包含了 worker 的描述信息，而 WorkerGroup 包含 worker 的集合信息。

```python
# torch/distributed/elastic/agent/server/api.py

@dataclass
class WorkerSpec:
    role: str
    local_world_size: int
    rdzv_handler: rdzv.RendezvousHandler
    fn: Optional[Callable] = None
    entrypoint: Union[Callable, str, None] = None
    args: tuple = ()
    max_restarts: int = 3
    monitor_interval: float = 0.1
    master_port: Optional[int] = None
    master_addr: Optional[str] = None
    local_addr: Optional[str] = None

class Worker:
    def __init__(
        self,
        local_rank: int,
        global_rank: int = -1,
        role_rank: int = -1,
        world_size: int = -1,
        role_world_size: int = -1,
    ):
        self.id: Any = None
        self.local_rank: int = local_rank
        self.global_rank: int = global_rank
        self.role_rank: int = role_rank
        self.world_size: int = world_size
        self.role_world_size: int = role_world_size

class WorkerGroup:
    def __init__(self, spec: WorkerSpec):
        self.spec = spec
        self.workers = [Worker(local_rank=i) for i in range(self.spec.local_world_size)]

        self.master_addr = None
        self.master_port = None
        self.state = WorkerState.INIT
```


## agent

从上述启动流程可以看到，agent 是启动的核心，`rendezvous` 和 `worker` 的定义都是传递给 agent，然后调用 agent 的 `run` 方法启动，这是一个阻塞函数，它代表了节点的生命周期，也即 torchrun 进程可以等同于 agent 进程。

`LocalElasticAgent` 中的 `run` 函数在父类 `SimpleElasticAgent` 中实现,

```python
# torch/distributed/elastic/agent/server/api.py 

class SimpleElasticAgent(ElasticAgent):
    def __init__(self, spec: WorkerSpec, exit_barrier_timeout: float = 300):
        self._worker_group = WorkerGroup(spec)

    def _rendezvous(self, worker_group: WorkerGroup) -> None:
        spec = worker_group.spec

        rdzv_info = spec.rdzv_handler.next_rendezvous()
        store = rdzv_info.store
        group_rank = rdzv_info.rank
        group_world_size = rdzv_info.world_size

        master_addr = spec.master_addr or rdzv_info.bootstrap_store_info.master_addr
        master_port = spec.master_port or rdzv_info.bootstrap_store_info.master_port

        self._store = store

        workers = self._assign_worker_ranks(
            store, group_rank, group_world_size, spec
        )
        worker_group.workers = workers
        worker_group.store = store
        worker_group.group_rank = group_rank
        worker_group.group_world_size = group_world_size
        worker_group.master_addr = master_addr
        worker_group.master_port = master_port


    def _assign_worker_ranks(
        self, store, group_rank: int, group_world_size: int, spec: WorkerSpec
    ) -> list[Worker]:
        base_role_rank = ...
        role_world_size = ...

        workers = []
        for local_rank in range(spec.local_world_size):
            worker = Worker(
                local_rank=local_rank,
                global_rank=base_global_rank + local_rank,
                role_rank=base_role_rank + local_rank,
                world_size=global_world_size,
                role_world_size=role_world_size,
            )
            workers.append(worker)
        return workers

    def _initialize_workers(self, worker_group: WorkerGroup) -> None:
        role = worker_group.spec.role

        self._rendezvous(worker_group)
        worker_ids = self._start_workers(worker_group)
        for local_rank, w_id in worker_ids.items():
            worker = worker_group.workers[local_rank]
            worker.id = w_id

        worker_group.state = WorkerState.HEALTHY

    def _restart_workers(self, worker_group: WorkerGroup) -> None:
        self._stop_workers(worker_group, is_restart=True)
        self._initialize_workers(worker_group)

    def run(self, role: str = DEFAULT_ROLE) -> RunResult:
        result = self._invoke_run(role)
        return result

    def _invoke_run(self, role: str = DEFAULT_ROLE) -> RunResult:
        spec = self._worker_group.spec
        role = spec.role

        self._initialize_workers(self._worker_group)
        rdzv_handler = spec.rdzv_handler

        while True:
            time.sleep(monitor_interval)
            run_result = self._monitor_workers(self._worker_group)
            state = run_result.state
            self._worker_group.state = state

            if state == WorkerState.SUCCEEDED:
                self._exit_barrier()
                return run_result
            elif state in {WorkerState.UNHEALTHY, WorkerState.FAILED}:
                self._remaining_restarts -= 1
                self._restart_workers(self._worker_group)
            elif state == WorkerState.HEALTHY:
                num_nodes_waiting = rdzv_handler.num_nodes_waiting()
                if num_nodes_waiting > 0:
                    self._restart_workers(self._worker_group)
```

`LocalElasticAgent` 中主要实现了 `_start_workers` 和 `_monitor_workers`, 这里和进程的封装 PContext 进行交互。

```python
# torch/distributed/elastic/agent/server/local_elastic_agent.py

class LocalElasticAgent(SimpleElasticAgent):
    def __init__(
        self,
        spec: WorkerSpec,
        start_method="spawn",
    ):
        super().__init__(spec, exit_barrier_timeout)
        self._start_method = start_method
        self._pcontext: Optional[PContext] = None
        self._rdzv_handler = spec.rdzv_handler

    def _start_workers(self, worker_group: WorkerGroup) -> dict[int, Any]:
        spec = worker_group.spec
        store = worker_group.store

        use_agent_store: bool = spec.rdzv_handler.use_agent_store

        args: dict[int, tuple] = {}
        envs: dict[int, dict[str, str]] = {}
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
                "MASTER_ADDR": worker_group.master_addr,
                "MASTER_PORT": str(worker_group.master_port),
                "TORCHELASTIC_RESTART_COUNT": str(restart_count),
                "TORCHELASTIC_MAX_RESTARTS": str(spec.max_restarts),
                "TORCHELASTIC_RUN_ID": spec.rdzv_handler.get_run_id(),
                "TORCHELASTIC_USE_AGENT_STORE": str(use_agent_store),
                "TORCH_NCCL_ASYNC_ERROR_HANDLING": os.getenv(
                    "TORCH_NCCL_ASYNC_ERROR_HANDLING", str(1)
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
            logs_specs=self._logs_specs,
            log_line_prefixes=log_line_prefixes,
            start_method=self._start_method,
        )
        return self._pcontext.pids()

    def _monitor_workers(self, worker_group: WorkerGroup) -> RunResult:
        result = self._pcontext.wait(0)
        if result:
            if result.is_failed():
                return RunResult(state=WorkerState.FAILED, failures=worker_failures)
            else:
                return RunResult(state=WorkerState.SUCCEEDED, return_values=workers_ret_vals)
        else:
            return RunResult(state=WorkerState.HEALTHY)
```


**启动流程**:

简化后的启动流程如下：

```bash
# launch_agent 中定义 进程的基础信息: 例如机器有 8 个 gpu，对应 8 个进程
spec = WorkerSpec(...) # launch_agent
    local_world_size=config.nproc_per_node,

# 根据 WorkerSpec 构建 WorkerGroup: 本机的 8 个进程抽象为 8 个 Worker，并组成 WorkerGroup
self._worker_group = WorkerGroup(spec) # agent init
    self.workers = [Worker(local_rank=i) for i in range(self.spec.local_world_size)]


# 根据 WorkerGroup 初始化 worker
self._initialize_workers(self._worker_group)
    # rendezvous 过程
    self._rendezvous(worker_group)
        # 通过对应的 rendezvous 模式获取共建信息: 协同分配 rank 的媒介
        rdzv_info = spec.rdzv_handler.next_rendezvous()
        # 根据全局信息为每个 worker 计算分配 rank
        workers = self._assign_worker_ranks(...)

    # 启动 workers
    worker_ids = self._start_workers(worker_group)
        # 为每个 worker 配置环境变量并启动进程
        for worker in worker_group.workers:
            local_rank = worker.local_rank
            worker_env = {
                "LOCAL_RANK": str(local_rank),
                "RANK": str(worker.global_rank),
                "GROUP_RANK": str(worker_group.group_rank),
            }
            envs[local_rank] = worker_env
            args[local_rank] = tuple(worker_args)

        self._pcontext = start_processes(...)
```

**rank 计算**:

* local_rank: 本节点内进程粒度的 rank
* global_rank: 全局进程粒度的 rank
* group_rank: 全局节点粒度的 rank

group_rank 计算方式:
```
global_rank = group_rank * group_world_size + local_rank
```

rank 计算的逻辑处理好累计问题其实比较简单，此处不详细展开。

**elastic**:

agent 的弹性能力体现在 `_invoke_run` 中，`_invoke_run` 会循环检测 worker 进程的状态:

* 如果 worker 进程正常退出则正常退出；
* 如果 worker 进程异常退出则重启 worker 进程；
* 如果 worker 进程正常但是有节点处于等待状态，即其他节点故障时会触发当前节点重启 worker 进程；

可以看出，agent 对当前节点上的 worker 进程负责，监控他们的健康状态，按需重启。

注意这里的查看状态函数 `_monitor_workers` 底层使用 timeout=0 的 poll 操作，所以是非阻塞的，而当前循环的等待是靠显示 sleep 实现的。

**为什么其他节点故障时会触发当前节点重启 worker 进程？**

当前架构中 agent-workers agent 是负责管理的进程，worker 是真正执行任务的进程，worker 之间还会通过 NCCL/gloo 等方式创建通信域进行通信从而可以交换数据。 当有节点故障时，当前逻辑是每个节点上的 agent 进程不退出，但是所有节点包括健康节点上的 worker 进程都会退出，再节点替换等逻辑恢复后，agent 重新拉起 worker 进程进而实现弹性。

这一实现的主要原因如下：

* 假设发生故障后，只有故障节点退出，健康节点的 worker 进程不退出，那么新 worker 启动后需要重新和已有进程建立新的通信域，这个过程的实现会极为复杂，远没有所有进程重启简单且稳定;
* 在 NCCL 信息域的角度看，peer 节点的异常是几乎无法感知的，无法感知就无法采取其他动作，并且不是处在所有状态的 OP  都是可撤销的（其实大多数是不可撤销的），即使利用超时等不可以的逻辑之上依然难以实现稳定的重建逻辑；
* 从 workflow 的角度看，worker 进程中的工作进程大概可以看作是计算、通信 OP 的串行序列，并没有一个 supervisor 的角色负责确认是否异常等上层逻辑，实现难度大且不够优雅。

以上原因导致主流的实现都使用 GPU 进程重启方式应对故障，实现容错和弹性。当然如果从探索角度看的话这已经不是一个新的话题，早几年就已经有相关的论文。


## rendezvous

`rendezvous`, 法语词，字面意思的约会，读音“夯dēi勿”， 用于分布式节点间协同，简单说就是节点间如何找到彼此，协商各自的 rank 等信息。

```python
# torch/distributed/elastic/rendezvous/__init__.py

from .registry import _register_default_handlers

_register_default_handlers()
```

可用的 rendezvous backend 是静态定义的，当前版本支持：`etcd`, `etcd-v2`, `c10d`, `static`，初始化化时注册到 `handler_registry` 中，通过 `rdzv_registry.get_rendezvous_handler` 获取对应的 handler.

```python
# torch/distributed/elastic/rendezvous/registry.py

def _register_default_handlers() -> None:
    handler_registry.register("etcd", _create_etcd_handler)
    handler_registry.register("etcd-v2", _create_etcd_v2_handler)
    handler_registry.register("c10d", _create_c10d_handler)
    handler_registry.register("static", _create_static_handler)

def _create_static_handler(params: RendezvousParameters) -> RendezvousHandler:
    from . import static_tcp_rendezvous

    return static_tcp_rendezvous.create_rdzv_handler(params)

def _create_c10d_handler(params: RendezvousParameters) -> RendezvousHandler:
    from .c10d_rendezvous_backend import create_backend

    backend, store = create_backend(params)

    return create_handler(store, backend, params)
```

这里主要看 `c10d` 的实现，`c10d` 的 tcp 版本通过 `TCPStore` 实现了 rendezvous，`TCPStore` 就是 pytorch 中重要的 kv 存储实现，在 `init_process_group` 等多个场景中都有使用。

```python
# torch/distributed/elastic/rendezvous/c10d_rendezvous_backend.py

def create_backend(params: RendezvousParameters) -> tuple[C10dRendezvousBackend, Store]:
    if store_type == "file":
        store = _create_file_store(params)
    elif store_type == "tcp":
        store = _create_tcp_store(params)
    backend = C10dRendezvousBackend(store, params.run_id)

    return backend, store

def _create_tcp_store(params: RendezvousParameters) -> TCPStore:
    host, port = parse_rendezvous_endpoint(params.endpoint, default_port=DEFAULT_PORT)

    store = TCPStore(
        host,
        port,
        is_master=is_server,
        multi_tenant=True,
        timeout=timedelta(seconds=read_timeout),
    )

    return store
```

划重点：用户参数中传递的 endpoint 对应的 host 和 port 会启动 `TCPStore` 服务端。

区别于 `static` backend, 使用 `c10d` 创建的 `rendezvous` 是动态 `DynamicRendezvousHandler`, 可以想见，它支持动态地进行节点协同，即在完成首次 rendezvous 后，可以动态的添加节点，删除节点，重新同步节点间的信息。

```python
# torch/distributed/elastic/rendezvous/dynamic_rendezvous.py

def create_handler(store: Store, backend: RendezvousBackend, params: RendezvousParameters) -> DynamicRendezvousHandler:
    return DynamicRendezvousHandler.from_backend(...)

class DynamicRendezvousHandler(RendezvousHandler):
    _node_desc_generator = _NodeDescGenerator()

    @classmethod
    def from_backend(...):
        node = cls._node_desc_generator.generate(local_addr)

        return cls(node, settings, backend.name, store, state_holder)

    def __init__(...):
        self._this_node = node
        self._bootstrap_store_info: Optional[RendezvousStoreInfo] = None

    def next_rendezvous(self) -> RendezvousInfo:
        try:
            rank, world_size = self._get_world()
            store = self._get_store()

        if os.getenv("TORCH_DISABLE_SHARE_RDZV_TCP_STORE", "0") == "1":
            bootstrap_store_info = RendezvousStoreInfo.build(
                rank, store, local_addr=self._this_node.addr
            )
            return RendezvousInfo(
                store,
                rank,
                world_size,
                bootstrap_store_info,
            )

        # This will only be hit when TCPStore sharing is enabled.
        if self._bootstrap_store_info is None:
            server_port = 0
            if rank == 0:
                self._shared_tcp_store_server = self._create_tcp_store_server(
                    self._this_node.addr, server_port
                )
                server_port = self._shared_tcp_store_server.port
            self._bootstrap_store_info = RendezvousStoreInfo.build(
                rank,
                store,
                local_addr=self._this_node.addr,
                server_port=server_port,  # For non-0 rank, this is a no-op
            )

        return RendezvousInfo(
            store,
            rank,
            world_size,
            self._bootstrap_store_info,  # type: ignore[assignment]
        )

class _NodeDescGenerator:
    def generate(self, local_addr: Optional[str] = None) -> _NodeDesc:
        return _NodeDesc(local_addr or socket.getfqdn(), os.getpid(), local_id)

```

可以看到 `rendezvous` 的结果通过 `RendezvousInfo` 进行了封装，其中包含了 `rank` 和 `world_size` 信息。

其中 RendezvousInfo 包含两个 TCPStore：

* `store` 是使用参数 rdzv endpoint 创建的 TCPStore;
* `_bootstrap_store_info` 中 master 存储了通过 store 交换回来的 addr 为 rank-0 地址，port 为 _create_tcp_store_server 创建的新的 TCPStore 的端口；

```python
# torch/distributed/elastic/rendezvous/api.py

@dataclass
class RendezvousStoreInfo:

    @staticmethod
    def build(
        rank: int,
        store: Store,
        local_addr: Optional[str],
        server_port: Optional[int] = None,
    ) -> "RendezvousStoreInfo":
        if rank == 0:
            addr = local_addr or socket.getfqdn()
            port = server_port or get_free_port()
            store.set(
                RendezvousStoreInfo.MASTER_ADDR_KEY,
                addr.encode(encoding="UTF-8"),  # type: ignore[arg-type]
            )
            store.set(
                RendezvousStoreInfo.MASTER_PORT_KEY,
                str(port).encode(encoding="UTF-8"),  # type: ignore[arg-type]
            )

        addr = store.get(RendezvousStoreInfo.MASTER_ADDR_KEY).decode(encoding="UTF-8")
        port = int(
            store.get(RendezvousStoreInfo.MASTER_PORT_KEY).decode(encoding="UTF-8")
        )
        return RendezvousStoreInfo(master_addr=addr, master_port=port)
```

* rank 为 0 的 “主节点” 会将自己的地址和端口信息存储到 `store` 中，所有节点会从 `store` 中获取新的 master 地址和端口信息即 rank 0 的信息存储在 RendezvousStoreInfo 中并返回；
* 每次执行都可能更新信息，每次调用 `next_rendezvous` 都会返回新的 `RendezvousInfo`，返回新的 `master` 地址和端口;
* 在弹性容错逻辑中，`_restart_workers` 会通过 `_initialize_workers` 调用 `_rendezvous` 来重新刷新 rank 等信息，RendezvousInfo 中的 master_addr/master_port 信息将会被使用；

# Reference

- [torchrun (Elastic Launch)](https://pytorch.org/docs/stable/elastic/run.html)

commit: 0da8127f77f9bf05ba204ea7659cb15ec85e88a7

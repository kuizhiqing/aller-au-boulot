# Overview

## BUILD

Ray 的编译使用 bazel，其中大量的 proto 编译也依赖于此，
具体定义在 `src/ray/protobuf/BUILD` 文件中，参考 [文档](https://rules-proto-grpc.com/en/latest/example.html).

## 启动部署

安装完成后，可以直接运行的相关命令如下

```python
# python/setup.py

entry_points={
    "console_scripts": [
        "ray=ray.scripts.scripts:main",
        "rllib=ray.rllib.scripts:cli [rllib]",
        "tune=ray.tune.scripts:cli",
        "ray-operator=ray.ray_operator.operator:main",
        "serve=ray.serve.scripts:cli",
    ]
}
```

常驻模式启动 ray 集群

```shell
# 启动 head 节点
ray start --head
# 启动 worker 节点
ray start --address=$RAY_HEAD_IP:6379

ray start --head --redis-password="" --port=6389
```

start 命令的解析如下

```python
# python/ray/scripts/scripts.py

# args 解析用的是 click

@cli.command()
@click.option("--head",...)
def start(...,head,...):
    ray_params = ray._private.parameter.RayParams(...)
    if head:
        # 启动 head 节点
        ray_params.update_if_absent(...)
        node = ray.node.Node(ray_params, head=True,...)
    else:
        # 启动 worker 节点
        bootstrap_address = services.canonicalize_bootstrap_address(address)
        ray_params.gcs_address = bootstrap_address
        node = ray.node.Node(ray_params, head=False,...)

cli.add_command(start)
def main():
    return cli()
```

可以看出本质上都是初始化了节点 Node 对象，是否为 head 则通过参数指定。

### Node

Node 节点的初始化

```python
# python/ray/node.py

class Node:
    def __init__(self, ray_params, head=False,...):
        self.head = head

        if not head:
            # GCS GRPC client, 确保 gcs 已启动
            self.get_gcs_client()

        # 初始化持久化存储
        storage._init_storage(ray_params.storage, is_head=head) 

        if head:
            self.validate_external_storage()

        if ...:
            # 启动 reaper 进程，负责在主进程意外退出后回收进程
            self.start_reaper_process()

        if head:
            self.start_head_processes()
            # 尝试写入 gcs
            self.get_gcs_client().internal_kv_put(...)

        if not connect_only:
            self.start_ray_processes()
            ray._private.services.wait_for_node(...)
```

Node 的初始化包括启动 start_head_processes 和 start_ray_processes 两部分。

```python
# python/ray/node.py

class Node:
    def start_head_processes(self):
        # 如果使用外部 redis，需要配置
        # 这里的逻辑目前有点 confuse，external 和 local 不够明确
        if self._ray_params.external_addresses is not None:
            self.start_or_configure_redis()
            self.create_redis_client()

        # 启动 gcs，包含 redis 服务, 默认端口 6379
        self.start_gcs_server()

        self.start_ray_client_server()

    def start_or_configure_redis(self):
        # 如果 external 有配置，并不真正启动
        ray._private.services.start_redis(...)

    def start_gcs_server(self):
        process_info = ray._private.services.start_gcs_server(self.redis_address,...)
        # 等待启动
        self.get_gcs_client()

    def start_ray_client_server(self):
        process_info = ray._private.services.start_ray_client_server(self.address, self._node_ip_address, ...)

    def start_ray_processes(self):
        # 启动节点上所有的进程
        self.destroy_external_storage()
        # 启动 raylet
        self.start_raylet(plasma_directory, object_store_memory)

    def start_raylet(self, ...):
        process_info = ray._private.services.start_raylet(
            self.redis_address,
            self.gcs_address,
            self._node_ip_address,
            ...)

```

* Head 节点启动 gcs 服务和 ray_client 服务
* Head 和 Worker 节点都启动 raylet 服务

这些服务的具体启动都被封装在 services 里。

### Services

Services 提供多种服务启动的封装，包括 redis 服务启动。

> 1.11 之前的版本 ray 通过启动 redis-server 二进制启动 redis，新版本中已经移除。

```python
# python/ray/_private/services.py

def start_gcs_server(redis_address, ...):
    # 调用 gcs 二进制启动服务，包含 redis 服务
    # GCS_SERVER_EXECUTABLE "core/src/ray/gcs/gcs_server"
    command = [GCS_SERVER_EXECUTABLE, "--redis_xxxx=", ...]
    process_info = start_ray_process(command, ...)

def start_ray_client_server(address, ray_client_server_ip,...):
    command = [
        sys.executable,    # python
        setup_worker_path, # ray/workers/setup_worker.py
        "-m",
        "ray.util.client.server",
        ...]
    process_info = start_ray_process(command, ...)

def start_raylet(redis_address, gcs_address, ...):
    # 启动 raylet，包括 local scheduler 和 object manager
    # 支持 python、java 和 cpp
    # RAYLET_EXECUTABLE "core/src/ray/raylet/raylet"
    command = [
        RAYLET_EXECUTABLE,
        f"--python_worker_command={subprocess.list2cmdline(start_worker_command)}",  # noqa
        f"--java_worker_command={subprocess.list2cmdline(java_worker_command)}",  # noqa
        f"--cpp_worker_command={subprocess.list2cmdline(cpp_worker_command)}",  # noqa
        ...]
    command.append("--agent_command={}".format(subprocess.list2cmdline(agent_command)))
    process_info = start_ray_process(command, ...)

def start_ray_process(command, ...):
    process = ConsolePopen(
        command,
        env=modified_env,
        cwd=cwd,
        stdout=stdout_file,
        stderr=stderr_file,
        stdin=subprocess.PIPE if pipe_stdin else None,
        preexec_fn=preexec_fn if sys.platform != "win32" else None,
        creationflags=CREATE_SUSPENDED if win32_fate_sharing else 0,
    )

class ConsolePopen(subprocess.Popen):
    pass

```

### setup worker & runtime env context

setup worker 的作用是执行不同的程序，

```python
# ray/workers/setup_worker.py

parser.add_argument("--serialized-runtime-env-context",...)
parser.add_argument("--language", ...)

if __name__ == "__main__":
    args, remaining_args = parser.parse_known_args()
    runtime_env_context = RuntimeEnvContext.deserialize(
        args.serialized_runtime_env_context or "{}"
    )
    runtime_env_context.exec_worker(remaining_args, Language.Value(args.language))
```

```python
# python/ray/_private/runtime_env/context.py

class RuntimeEnvContext:
    def exec_worker(self, passthrough_args: List[str], language: Language):
        os.environ.update(self.env_vars)

        # exec [python] passthrough_args
        command_str = " && ".join(...)

        if sys.platform == "win32":
            os.system(command_str)
        else:
            os.execvp("bash", args=["bash", "-c", command_str])
```

### client server

```bash
python -m ray.util.client.server --address=x.x.x.x:6379 --host=0.0.0.0 --port=10001 --mode=proxy --redis-password=
```

```python
# python/ray/util/client/server/__main__.py
# -> server.py main
# python/ray/util/client/server/server.py

def main():
    server = serve(hostport, ray_connect_handler)
    while True:
        ray.experimental.internal_kv._internal_kv_put(..., HEALTHCHECK)

def serve(connection_str, ray_connect_handler=None):
    server = grpc.server(
        futures.ThreadPoolExecutor(
            max_workers=CLIENT_SERVER_MAX_THREADS,
            thread_name_prefix="ray_client_server",
        ),
    )
    # mode proxy
    task_servicer = RayletServicerProxy(None, proxy_manager)
    data_servicer = DataServicerProxy(proxy_manager)
    logs_servicer = LogstreamServicerProxy(proxy_manager)
    # else
    task_servicer = RayletServicer(ray_connect_handler)
    data_servicer = DataServicer(task_servicer)
    logs_servicer = LogstreamServicer()
    ray_client_pb2_grpc.add_RayletDriverServicer_to_server(task_servicer, server)
    ray_client_pb2_grpc.add_RayletDataStreamerServicer_to_server(data_servicer, server)
    ray_client_pb2_grpc.add_RayletLogStreamerServicer_to_server(logs_servicer, server)
    server.start()
```

```python
# python/ray/util/client/server/server.py

# class RayletServicerProxy(ray_client_pb2_grpc.RayletDriverServicer):
class RayletServicer(ray_client_pb2_grpc.RayletDriverServicer):
    def KVPut(self, request, context=None) -> ray_client_pb2.KVPutResponse:
    def KVGet(self, request, context=None) -> ray_client_pb2.KVGetResponse:
    def KVDel(self, request, context=None) -> ray_client_pb2.KVDelResponse:
    def KVList(self, request, context=None) -> ray_client_pb2.KVListResponse:
    def ListNamedActors(...):
    def ClusterInfo(self, request, context=None) -> ray_client_pb2.ClusterInfoResponse:
    def release(self, client_id: str, id: bytes) -> bool:
    def release_all(self, client_id):
    def Terminate(self, req, context=None):
    def GetObject(self, request: ray_client_pb2.GetRequest, context):
    def PutObject( self, request: ray_client_pb2.PutRequest, context=None) -> ray_client_pb2.PutResponse:
    def WaitObject(self, request, context=None) -> ray_client_pb2.WaitResponse:
    def Schedule( self, task: ray_client_pb2.ClientTask, context=None) -> ray_client_pb2.ClientTaskTicket:
    def lookup_or_register_func( self, id: bytes, client_id: str, options: Optional[Dict]) -> ray.remote_function.RemoteFunction:
    def lookup_or_register_actor( self, id: bytes, client_id: str, options: Optional[Dict]):
    def unify_and_track_outputs(self, output, client_id):
```

### 总结

`ray start [--head]`, 将启动以下进程

```shell
# 以下进程只在 head 节点运行
# GCS_SERVER_EXECUTABLE 
/usr/local/lib/python3.7/dist-packages/ray/core/src/ray/gcs/gcs_server 
/usr/bin/python3.7 -m ray.util.client.server --host=0.0.0.0 --port=10001 --mode=proxy

# worker 进程
# RAYLET_EXECUTABLE
/usr/local/lib/python3.7/dist-packages/ray/core/src/ray/raylet/raylet
/usr/bin/python3.7 -u /usr/local/lib/python3.7/dist-packages/ray/dashboard/agent.py

/usr/bin/python3.7 -u /usr/local/lib/python3.7/dist-packages/ray/autoscaler/_private/monitor.py
/usr/bin/python3.7 -u /usr/local/lib/python3.7/dist-packages/ray/dashboard/dashboard.py
/usr/bin/python3.7 -u /usr/local/lib/python3.7/dist-packages/ray/_private/log_monitor.py
```

```shell
# 新版本中的以下进程已被移除
/usr/local/lib/python3.7/dist-packages/ray/core/src/ray/thirdparty/redis/src/redis-server *:6379
/usr/local/lib/python3.7/dist-packages/ray/core/src/ray/thirdparty/redis/src/redis-server *:64712
```

```shell
# java worker command
python ray/workers/setup_worker.py java -Dx=x -cp xx RAY_WORKER_DYNAMIC_OPTION_PLACEHOLDER io.ray.runtime.runner.worker.DefaultWorker

#  cpp worker command
cpp/default_worker
```

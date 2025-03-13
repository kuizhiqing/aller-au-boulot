# architecture


## online serving

服务启动命令

```bash
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
```

命令在 `pyproject.toml` 中定义

```toml
[project.scripts]
vllm = "vllm.entrypoints.cli.main:main"
```

入口
`vllm/entrypoints/cli/main.py`
dispatch 到真实启动的入口
`vllm.entrypoints.cli.serve`.

```python
# vllm/entrypoints/cli/serve.py

class ServeSubcommand(CLISubcommand):
    def __init__(self):
        self.name = "serve"

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        args.model = args.model_tag
        uvloop.run(run_server(args))

    def subparser_init(self, subparsers):
        serve_parser = subparsers.add_parser("serve", usage="vllm serve <model_tag> [options]")
        serve_parser.add_argument("model_tag", ...)
        serve_parser.add_argument("--config", ...) # YAML config file
        return make_arg_parser(serve_parser)
```

可以看到，`serve` 命令的入口是 `vllm.entrypoints.cli.serve.ServeSubcommand.cmd`，它调用 `vllm.entrypoints.cli.serve.run_server`，而 `run_server` 会创建 `uvicorn` 服务。

```python
# vllm/entrypoints/openai/api_server.py

@asynccontextmanager
async def build_async_engine_client(args) -> AsyncIterator[EngineClient]:
    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine_client = AsyncLLMEngine.from_engine_args(engine_args, ...)
    yield engine_client

def build_app(args: Namespace) -> FastAPI:
    app = FastAPI(lifespan=lifespan)
    app.include_router(router)
    return app

async def run_server(args, **uvicorn_kwargs) -> None:
    async with build_async_engine_client(args) as engine_client:
        app = build_app(args)
        model_config = await engine_client.get_model_config()
        await init_app_state(engine_client, model_config, app.state, args)
        await serve_http(app, ...  **uvicorn_kwargs)


if __name__ == "__main__":
    parser = FlexibleArgumentParser(...)
    args = parser.parse_args()
    uvloop.run(run_server(args))
```

`run_server` 是个异步函数，它完成了 2 个主要任务：

* 创建异步 engine : `AsyncLLMEngine`
* 创建 FastAPI 应用，承接 http 请求

这里通过 `init_app_state` 初始化了 app.state，即把 vllm LLMEngine 设置给了 FastAPI。

其中 `serve_http` 通过 uvicorn 实现了 http 服务的启动。

```python
# vllm/entrypoints/launcher.py

async def serve_http(app: FastAPI, sock: Optional[socket.socket], **uvicorn_kwargs: Any):
    config = uvicorn.Config(app, **uvicorn_kwargs)
    server = uvicorn.Server(config)
    loop = asyncio.get_running_loop()
    server_task = loop.create_task(
        server.serve(sockets=[sock] if sock else None))
```

同时可以看到 `vllm serve` 的等价启动方式是

```bash
python3 -m vllm.entrypoints.openai.api_server --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
```

## offline inference

```python
from vllm import LLM

prompts = "Hello, my name is"

llm = LLM(model="facebook/opt-125m")

outputs = llm.generate(prompts)
```

```python
# vllm/__init__.py

from vllm.entrypoints.llm import LLM
```  

LLM

```python
# vllm/entrypoints/llm.py

class LLM:
    def __init__(self, model, ...) -> None:
        worker_cls = kwargs["worker_cls"]
        engine_args = EngineArgs(model, task, tokenizer)
        self.engine_class = self.get_engine_class()
        self.llm_engine = self.engine_class.from_engine_args(
            engine_args, usage_context=UsageContext.LLM_CLASS)

    @staticmethod
    def get_engine_class() -> Type[LLMEngine]:
        if envs.VLLM_USE_V1:
            from vllm.v1.engine.llm_engine import LLMEngine as V1LLMEngine
            return V1LLMEngine
        return LLMEngine

    def get_tokenizer(self) -> AnyTokenizer:
        return self.llm_engine.get_tokenizer_group(TokenizerGroup).tokenizer

    @overload
    def generate(self, prompts, sampling_params):
        outputs = self._run_engine(use_tqdm=use_tqdm)
        return self.engine_class.validate_outputs(outputs, RequestOutput)

    def collective_rpc(self, ...):
        executor = self.llm_engine.model_executor
        return executor.collective_rpc(method, timeout, args, kwargs)

    def apply_model(self, func: Callable[[nn.Module], _R]) -> list[_R]:
        executor = self.llm_engine.model_executor
        return executor.apply_model(func)
```

## Engine

## AsyncLLMEngine

```python
# vllm/engine/async_llm_engine.py

if envs.VLLM_USE_V1:
    from vllm.v1.engine.async_llm import AsyncLLM
    AsyncLLMEngine = AsyncLLM
```

```python
# vllm/v1/engine/async_llm.py

class AsyncLLM(EngineClient):

    def __init__(self, vllm_config, executor_class):
        # Tokenizer (+ ensure liveness if running in another process).
        self.tokenizer = init_tokenizer_from_configs(...)
        self.tokenizer.ping()

        # Processor (converts Inputs --> EngineCoreRequests).
        self.processor = Processor(..., tokenizer...)

        # OutputProcessor (converts EngineCoreOutputs --> RequestOutput).
        self.output_processor = OutputProcessor(self.tokenizer, ...)

        # EngineCore (starts the engine in background process).
        self.engine_core = EngineCoreClient.make_client(...)

        self.output_handler: Optional[asyncio.Task] = None

    @classmethod
    def from_engine_args(cls, ...) -> "AsyncLLM":
        executor_class = Executor.get_class(vllm_config)
        return cls(...)

    async def add_request(self, request_id, prompt, params, ...) -> asyncio.Queue[RequestOutput]:
        # 1) Create a new output queue for the request.
        queue: asyncio.Queue[RequestOutput] = asyncio.Queue()

        # 2) Convert Input --> Request.
        request = self.processor.process_inputs(request_id, prompt, params, ...)

        # 3) Add the request to OutputProcessor (this process).
        self.output_processor.add_request(request, queue)

        # 4) Add the EngineCoreRequest to EngineCore (separate process).
        await self.engine_core.add_request_async(request)

        return queue

    async def generate(self, prompt, sampling_params, ...) -> AsyncGenerator[RequestOutput, None]:
        """
        Main function called by the API server to kick off a request
            * 1) Making an AsyncStream corresponding to the Request.
            * 2) Processing the Input.
            * 3) Adding the Request to the Detokenizer.
            * 4) Adding the Request to the EngineCore (separate process).

        A separate output_handler loop runs in a background AsyncIO task, 
        pulling outputs from EngineCore and putting them into the 
        per-request AsyncStream.

        The caller of generate() iterates the returned AsyncGenerator,
        returning the RequestOutput back to the caller.
        """

        try:
            # We start the output_handler on the first call to generate() so
            # we can call __init__ before the event loop, which enables us
            # to handle startup failure gracefully in the OpenAI server.
            self.output_handler = asyncio.create_task(self._run_output_handler())

            q = await self.add_request(request_id, prompt, sampling_params, ...)

            # The output_handler task pushes items into the queue.
            # This task pulls from the queue and yields to caller.
            finished = False
            while not finished:
                out = q.get_nowait() if not q.empty() else await q.get()

                while not q.empty():
                    next_out = q.get_nowait()
                    if sampling_params.output_kind == RequestOutputKind.DELTA:
                        out.add(next_out)
                    else:
                        out = next_out

                finished = out.finished
                yield out

    async def _run_output_handler(self):
        """Background loop: pulls from EngineCore and pushes to AsyncStreams."""

        try:
            while True:
                # 1) Pull EngineCoreOutputs from the EngineCore.
                outputs = await self.engine_core.get_output_async()

                iteration_stats = IterationStats() if self.log_stats else None

                # Split outputs into chunks of at most
                # VLLM_V1_OUTPUT_PROC_CHUNK_SIZE, so that we don't block the
                # event loop for too long.
                num_outputs = len(outputs.outputs)
                if num_outputs <= VLLM_V1_OUTPUT_PROC_CHUNK_SIZE:
                    slices = (outputs.outputs, )
                else:
                    slices = np.array_split(
                        outputs.outputs,
                        cdiv(num_outputs, VLLM_V1_OUTPUT_PROC_CHUNK_SIZE))

                for i, outputs_slice in enumerate(slices):
                    # 2) Process EngineCoreOutputs.
                    processed_outputs = self.output_processor.process_outputs(
                        outputs_slice, outputs.timestamp, iteration_stats)
                    # NOTE: RequestOutputs are pushed to their queues.
                    assert not processed_outputs.request_outputs

                    # Allow other asyncio tasks to run between chunks
                    if i + 1 < len(slices):
                        await asyncio.sleep(0)

                    # 3) Abort any reqs that finished due to stop strings.
                    await self.engine_core.abort_requests_async(
                        processed_outputs.reqs_to_abort)

                # 4) Logging.
                # TODO(rob): make into a coroutine and launch it in
                # background thread once Prometheus overhead is non-trivial.
                self._log_stats(
                    scheduler_stats=outputs.scheduler_stats,
                    iteration_stats=iteration_stats,
                )
```

## LLMEngine


## Entrypoint
## Excutor

Executor

```python
# vllm/v1/executor/abstract.py

class Executor(ExecutorBase):
    @staticmethod
    def get_class(vllm_config: VllmConfig) -> Type["Executor"]:
        executor_class = RayDistributedExecutor
        executor_class = MultiprocExecutor
        executor_class = UniProcExecutor
        executor_class = ExecutorWithExternalLauncher
        return executor_class

    def initialize(self, kv_cache_configs: List[KVCacheConfig]) -> None:
        self.collective_rpc("initialize_cache", args=(kv_cache_configs, ))
        self.collective_rpc("compile_or_warm_up_model")

    def execute_model(
        self,
        scheduler_output,
    ) -> Union[ModelRunnerOutput, Future[ModelRunnerOutput]]:
        output = self.collective_rpc("execute_model", args=(scheduler_output, ))
        return output[0]

```

```python
# vllm/executor/executor_base.py
class ExecutorBase(ABC):
    def __init__(self, vllm_config: VllmConfig) -> None:
        self._init_executor()
```

MultiprocExecutor

```python
# vllm/v1/executor/multiproc_executor.py

class MultiprocExecutor(Executor):
	def _init_executor(self) -> None:
        self.rpc_broadcast_mq = MessageQueue(self.world_size, self.world_size)
        scheduler_output_handle = self.rpc_broadcast_mq.export_handle()

        self.workers: List[WorkerProcHandle] = []
        for rank in range(self.world_size):
            worker = WorkerProc.make_worker_process(self.vllm_config, ...)
            self.workers.append(worker)

class WorkerProc:
    def __init__(self, vllm_config, local_rank, rank, ...):
		wrapper = WorkerWrapperBase(vllm_config=vllm_config, ...)
		wrapper.init_worker(all_kwargs)
		self.worker = wrapper.worker

		self.rpc_broadcast_mq = MessageQueue.create_from_handle(...)
		self.worker_response_mq = MessageQueue(1, 1)

		self.worker.init_device()
		self.worker.load_model()

    @staticmethod
    def make_worker_process(vllm_config, ...) -> WorkerProcHandle:
        proc = context.Process(target=WorkerProc.worker_main, ..., daemon=True)
        proc.start()
        return WorkerProcHandle(proc, rank, ready_path, worker_response_mq)

    @staticmethod
    def worker_main(*args, **kwargs):
        try:
            worker = WorkerProc(*args, **kwargs)
            worker.worker_busy_loop()

    def worker_busy_loop(self):
        while True:
            method, args, kwargs = self.rpc_broadcast_mq.dequeue()
            func = getattr(self.worker, method)
            output = func(*args, **kwargs)
            self.worker_response_mq.enqueue((SUCCESS, output))
```

RayDistributedExecutor

```python
# vllm/v1/executor/ray_distributed_executor.py

class RayDistributedExecutor(RayDistributedExecutorV0, Executor):
    def execute_model(
        self,
        scheduler_output,
    ) -> Union[ModelRunnerOutput, Future[ModelRunnerOutput]]:
        self.forward_dag = self._compiled_ray_dag(enable_asyncio=False)
        refs = self.forward_dag.execute(scheduler_output)
        return refs[0].get()
```

UniProcExecutor
ExecutorWithExternalLauncher

```python
# vllm/executor/uniproc_executor.py

class UniProcExecutor(ExecutorBase):
    def _init_executor(self) -> None:
        self.driver_worker = WorkerWrapperBase(vllm_config=self.vllm_config, rpc_rank=0)
        self.collective_rpc("init_worker", args=([kwargs], ))
        self.collective_rpc("init_device")
        self.collective_rpc("load_model")

    def collective_rpc(self,
                       method: Union[str, Callable],
                       timeout: Optional[float] = None,
                       args: Tuple = (),
                       kwargs: Optional[Dict] = None) -> List[Any]:
        answer = run_method(self.driver_worker, method, args, kwargs)
        return [answer]

class ExecutorWithExternalLauncher(UniProcExecutor):
```

WorkerWrapperBase

    """
    This class represents one process in an executor/engine. It is responsible
    for lazily initializing the worker and handling the worker's lifecycle.
    We first instantiate the WorkerWrapper, which remembers the worker module
    and class name. Then, when we call `update_environment_variables`, and the
    real initialization happens in `init_worker`.
    """

        """
        Initialize the worker wrapper with the given vllm_config and rpc_rank.
        Note: rpc_rank is the rank of the worker in the executor. In most cases,
        it is also the rank of the worker in the distributed group. However,
        when multiple executors work together, they can be different.
        e.g. in the case of SPMD-style offline inference with TP=2,
        users can launch 2 engines/executors, each with only 1 worker.
        All workers have rpc_rank=0, but they have different ranks in the TP
        group.
        """

```python
# vllm/worker/worker_base.py

class WorkerWrapperBase:
    def __init__(self, vllm_config, rpc_rank) -> None:
        self.worker: Optional[WorkerBase] = None
        init_cached_hf_modules()

    def init_worker(self, all_kwargs: List[Dict[str, Any]]) -> None:
        worker_class = resolve_obj_by_qualname(self.vllm_config.parallel_config.worker_cls)
        worker_class = cloudpickle.loads(self.vllm_config.parallel_config.worker_cls)
        self.worker = worker_class(**kwargs)

    def execute_method(self, method: Union[str, bytes], *args, **kwargs):
        return run_method(target, method, args, kwargs)

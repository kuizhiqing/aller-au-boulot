# architecture

## online serving

**TL;DR;**

* 使用 vllm serve 命令启动 FastAPI/HTTP 模型服务，支持 OpenAI 协议访问；
* 本质上创建了 AsyncLLMEngine，当访问 `/v1/chat/completions` 接口时，会调用 `engine.generate` 方法；

使用命令

```bash
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
```

可以在本地启动一个基于 `DeepSeek-R1-Distill-Qwen-7B` 的模型服务。

服务的默认端口是 8000，采用 [OpenAI 的协议](https://platform.openai.com/docs/api-reference/chat/create)
格式，可以使用 http 或 [openai sdk]() 等方式访问。

**curl 访问**

```python
curl https://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer EMPTY" \
  -d '{
    "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": "Hello!"
      }
    ]
  }'
```

**openai sdk 访问**

```python
from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1",
)

chat_completion = client.chat.completions.create(
    messages=[{
        "role": "system",
        "content": "You are a helpful assistant."
    }, {
        "role": "user",
        "content": "Hello!"
    }],
    model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
)

print("Chat completion results:")
print(chat_completion)
```

**entrypoint**

`vllm` 命令在安装 vllm 时被安装，在 `pyproject.toml` 中定义

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

**请求链路**

```python
# vllm/entrypoints/openai/api_server.py

router = APIRouter()

@router.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest, raw_request: Request):
    handler = chat(raw_request)
    generator = await handler.create_chat_completion(request, raw_request)
    if isinstance(generator, ChatCompletionResponse):
        return JSONResponse(content=generator.model_dump())

    return StreamingResponse(content=generator, media_type="text/event-stream")

def chat(request: Request) -> Optional[OpenAIServingChat]:
    return request.app.state.openai_serving_chat


async def init_app_state(engine_client, model_config, state, args):
    state.openai_serving_chat = OpenAIServingChat(
        engine_client, model_config, state.openai_serving_models, ...
    )
```

请求 `/v1/chat/completions` 接口时，会调用 `OpenAIServingChat` 的 `create_chat_completion` 方法。

```python
# vllm/entrypoints/openai/serving_chat.py

class OpenAIServingChat(OpenAIServing):

    async def create_chat_completion(self, ChatCompletionRequest, Request):

        tokenizer = await self.engine_client.get_tokenizer(lora_request)

        (conversation, request_prompts, engine_prompts) = 
            await self._preprocess_chat(request, tokenizer, request.messages, ...)

        for i, engine_prompt in enumerate(engine_prompts):
            sampling_params = request.to_sampling_params(...)

            generator = self.engine_client.generate(engine_prompt, sampling_params, request_id,)

            generators.append(generator)

        result_generator, = generators

        return await self.chat_completion_full_generator( # chat_completion_stream_generator
            request, result_generator, request_id, model_name,
            conversation, tokenizer, request_metadata)

    async def chat_completion_full_generator(self, request, result_generator, ...):
        async for res in result_generator:
            final_res = res

        role = self.get_chat_request_role(request)

        for output in final_res.outputs:
            logprobs = self._create_chat_logprobs(...)

            message = ChatMessage(role=role, content=output.text)

            choice_data = ChatCompletionResponseChoice(
                index=output.index,
                message=message,
                logprobs=logprobs, ...)
            choices.append(choice_data)

        response = ChatCompletionResponse(
            choices=choices,
            prompt_logprobs=final_res.prompt_logprobs,
        )

        return response
```

可以看到 `create_chat_completion` 的主要逻辑是调用 engine 的 `generate` 方法，生成结果，
`prompt` 由 `OpenAIServing._preprocess_chat` 进行预处理，包括 template 化和 tokenization。

```python
# vllm/entrypoints/openai/serving_engine.py

class OpenAIServing:

    async def _preprocess_chat(...):
        conversation, mm_data_future = parse_chat_messages_futures(...)

        request_prompt = apply_hf_chat_template(tokenizer, conversation=conversation,)

        prompt_inputs = await self._tokenize_prompt_input_async(request, tokenizer, request_prompt,)

        engine_prompt = TokensPrompt(prompt_token_ids=prompt_inputs["prompt_token_ids"])

        return conversation, [request_prompt], [engine_prompt]
```

注意 `request_prompt -> prompt_inputs -> engine_prompt` 的过程，engine_prompt 中包含了完整 prompt 信息。


函数 `generate` 调用的 3 个参数

* prompt : 未 tokenized 的格式化输入；
* sampling_params : 采样参数，包括 temperature, top_p, top_k 等；
* request_id : 请求 id，用于区分不同请求。

注意到

* 生成的文本位于 `response.choices.message.content` 中。
* generate 返回的 generator 中的 `output.text` 是已经经过 tokenizer decode 的。

根据是否 stream 请求，调用 chat_completion_stream_generator 和 chat_completion_full_generator 返回不同的结果。


推理实现的核心逻辑 `engine.generate` 将在 Engine 部分分析。


## offline inference

与 serving 模式不同，离线推理模式直接在程序内加载模型通过函数调用的方式得到推理结果。

**demo**

```python
from vllm import LLM

prompts = "Hello, my name is"

llm = LLM(model="facebook/opt-125m")

outputs = llm.generate(prompts)

for output in outputs:
    generated_text = output.outputs[0].text
```

从 demo 中可以看出流程为使用 `LLM` 类，调用 `generate` 方法，得到结果。

```python
# vllm/__init__.py

from vllm.entrypoints.llm import LLM
```  

与在线不同，离线使用的 `LLM` 类是对应的是同步 `LLMEngine`, 也即在线 serving 通常使用异步模式，离线 inference 模式通常使用同步模式。


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

    @overload
    def generate(self, prompts, sampling_params):
        self._validate_and_add_requests(prompts=parsed_prompts, params=sampling_params,...)
        outputs = self._run_engine(use_tqdm=use_tqdm)
        return self.engine_class.validate_outputs(outputs, RequestOutput)

    def _validate_and_add_requests(self, prompts, ...):
        for i, prompt in enumerate(prompts):
            self._add_request(prompt, ...)

    def _add_request(self, prompt, ...):
        self.llm_engine.add_request( request_id, prompt, params, ...)

    def _run_engine(self, *, use_tqdm: bool) -> List[Union[RequestOutput, PoolingRequestOutput]]:
        while self.llm_engine.has_unfinished_requests():
            step_outputs = self.llm_engine.step()
            for output in step_outputs:
                if output.finished:
                    outputs.append(output)

        return sorted(outputs, key=lambda x: int(x.request_id))
```

总结 LLM 的流程为，

1. 初始化时创建 `LLMEngine` 实例;
2. 调用 `generate` 时首先调用 `engine.add_request` 方法添加请求;
3. 再调用 `engine.step` 方法执行请求，获取推理结果；

具体如何获取推理结果，需要看 `LLMEngine` 的实现。

## Engine

## AsyncLLMEngine

V1 版本的 AsyncLLMEngine 定义于 `vllm.v1.engine.async_llm` 的 AsyncLLM

```python
# vllm/engine/async_llm_engine.py

if envs.VLLM_USE_V1:
    from vllm.v1.engine.async_llm import AsyncLLM
    AsyncLLMEngine = AsyncLLM
```

AsyncLLM 定义一个异步的 LLM，其关键代码如下，

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
        self.engine_core = EngineCoreClient.make_client(..., vllm_config, executor_class, ...)

        self.output_handler: Optional[asyncio.Task] = None

    @classmethod
    def from_engine_args(cls, ...) -> "AsyncLLM":
        executor_class = Executor.get_class(vllm_config)
        return cls(vllm_config, executor_class, ...)

    async def generate(self, prompt, sampling_params, ...) -> AsyncGenerator[RequestOutput, None]:
        self.output_handler = asyncio.create_task(self._run_output_handler())

        q = await self.add_request(request_id, prompt, sampling_params, ...)

        while not finished:
            out = q.get_nowait() if not q.empty() else await q.get()
            yield out

    async def _run_output_handler(self):
        """Background loop: pulls from EngineCore and pushes to AsyncStreams."""

        while True:
            # 1) Pull EngineCoreOutputs from the EngineCore.
            outputs = await self.engine_core.get_output_async()

            slices = (outputs.outputs, )

            for i, outputs_slice in enumerate(slices):
                # 2) Process EngineCoreOutputs.
                processed_outputs = self.output_processor.process_outputs(
                    outputs_slice, outputs.timestamp, iteration_stats)

                # 3) Abort any reqs that finished due to stop strings.
                await self.engine_core.abort_requests_async(
                    processed_outputs.reqs_to_abort)

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
```

分析 `generate()` 调用流程

```
engine_core.get_output_async()
output_processor.process_outputs()
engine_core.abort_requests_async()

processor.process_inputs()
output_processor.add_request()
engine_core.add_request_async()
```

AsyncLLM (V1 版本的 AsyncLLMEngine) 是一个异步的 LLM，系统的核心组件，它的初始化过程启动了主要组件：

* EngineCoreClient + Executor
* Processor + Tokenizer
* OutputProcessor + Tokenizer


```
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
```

## EngineCoreClient

```
engine_core.get_output_async()
engine_core.abort_requests_async()
engine_core.add_request_async()
```

```python
# vllm/v1/engine/core_client.py

class EngineCoreClient(ABC):
    @staticmethod
    def make_client(..., vllm_config, executor_class, ...) -> "EngineCoreClient":
        if multiprocess_mode and asyncio_mode:
            return AsyncMPClient(vllm_config, executor_class, log_stats)

        if multiprocess_mode and not asyncio_mode:
            return SyncMPClient(vllm_config, executor_class, log_stats)

        return InprocClient(vllm_config, executor_class, log_stats)


class InprocClient(EngineCoreClient):
    def __init__(self, *args, **kwargs):
        self.engine_core = EngineCore(*args, **kwargs)

    def get_output(self) -> EngineCoreOutputs:
        return self.engine_core.step()

    def add_request(self, request: EngineCoreRequest) -> None:
        self.engine_core.add_request(request)

    def abort_requests(self, request_ids: List[str]) -> None:
        if len(request_ids) > 0:
            self.engine_core.abort_requests(request_ids)


class MPClient(EngineCoreClient):
    def __init__(..., vllm_config, executor_class, ...):
        self.ctx =  zmq.asyncio.Context()
        self.output_socket = make_zmq_socket(self.ctx, output_path, zmq.constants.PULL)
        self.input_socket = make_zmq_socket(self.ctx, input_path, zmq.constants.PUSH)

        self.proc_handle = BackgroundProcHandle(
            input_path=input_path,
            output_path=output_path,
            target_fn=EngineCoreProc.run_engine_core,
            process_kwargs={"vllm_config": vllm_config, "executor_class": executor_class, })


class SyncMPClient(MPClient):
    ...

class AsyncMPClient(MPClient):
    ...
```

EngineCoreClient 的 3 种实现 EngineCore 

* InprocClient: In process EngineCore (for V0-style LLMEngine use)
* SyncMPClient: ZMQ + background proc EngineCore (for LLM)
* AsyncMPClient: ZMQ + background proc EngineCore w/ asyncio (for AsyncLLM)

核心实现逻辑由 BackgroundProcHandle 启动 `EngineCoreProc.run_engine_core` 来实现。

```python

class SyncMPClient(MPClient):
    def __init__(self, vllm_config, executor_class, ...):
        def process_outputs_socket():
            while True:
                (frame, ) = output_socket.recv_multipart(copy=False)
                outputs = decoder.decode(frame.buffer)
                outputs_queue.put_nowait(outputs)
        Thread(target=process_outputs_socket, daemon=True).start()

    def get_output(self) -> EngineCoreOutputs:
        return self.outputs_queue.get()

    def _send_input(self, request_type, request) -> None:
        msg = (request_type.value, self.encoder.encode(request))
        self.input_socket.send_multipart(msg, copy=False)

    def add_request(self, request: EngineCoreRequest) -> None:
        self._send_input(EngineCoreRequestType.ADD, request)

    def abort_requests(self, request_ids: List[str]) -> None:
        self._send_input(EngineCoreRequestType.ABORT, request_ids)
```

```python
class AsyncMPClient(MPClient):
    async def _start_output_queue_task(self):
        self.outputs_queue = asyncio.Queue()

        async def process_outputs_socket():
            while True:
                (frame, ) = await output_socket.recv_multipart(copy=False)
                outputs: EngineCoreOutputs = decoder.decode(frame.buffer)
                outputs_queue.put_nowait(outputs)

        self.queue_task = asyncio.create_task(process_outputs_socket())

    async def get_output_async(self) -> EngineCoreOutputs:
        await self._start_output_queue_task()
        return await self.outputs_queue.get()

    async def _send_input(request_type, request):
        msg = (request_type.value, self.encoder.encode(request))
        await self.input_socket.send_multipart(msg, copy=False)

    async def add_request_async(self, request: EngineCoreRequest) -> None:
        await self._send_input(EngineCoreRequestType.ADD, request)

    async def abort_requests_async(self, request_ids: List[str]) -> None:
        await self._send_input(EngineCoreRequestType.ABORT, request_ids)

```

SyncMPClient/AsyncMPClient 的实现

1. `add_request/add_request_async` 通过 input_socket 向 mq 中发送消息；
2. while 循环从 output_socket 中获取消息，放入 outputs_queue 中；
3. `get_output/get_output_async` 从 outputs_queue 中得到消息；


```python
# vllm/v1/engine/core.py

class EngineCore:
    def __init__(self, vllm_config, executor_class):
        self.model_executor = executor_class(vllm_config)

        num_gpu_blocks, num_cpu_blocks = self._initialize_kv_caches(vllm_config)

        self.scheduler = Scheduler(scheduler_config, model_config, cache_config, ...)

        self.batch_queue_size = self.model_executor.max_concurrent_batches

        if self.batch_queue_size > 1:
            self.batch_queue = queue.Queue(self.batch_queue_size)

    def _initialize_kv_caches(self, vllm_config: VllmConfig) -> Tuple[int, int]:
        self.model_executor.initialize(kv_cache_configs)
        return num_gpu_blocks, num_cpu_blocks

    def add_request(self, request: EngineCoreRequest):
        req = Request.from_engine_core_request(request)

        self.scheduler.add_request(req)

    def abort_requests(self, request_ids: List[str]):
        self.scheduler.finish_requests(request_ids, RequestStatus.FINISHED_ABORTED)

    def step(self) -> EngineCoreOutputs:
        scheduler_output = self.scheduler.schedule()
        output = self.model_executor.execute_model(scheduler_output)
        engine_core_outputs = self.scheduler.update_from_output(scheduler_output, output)
        return engine_core_outputs

    def step_with_batch_queue(self) -> Optional[EngineCoreOutputs]:
        scheduler_output = self.scheduler.schedule()
        future = self.model_executor.execute_model(scheduler_output)
        self.batch_queue.put_nowait( (future, scheduler_output))

        future, scheduler_output = self.batch_queue.get(timeout=POLLING_TIMEOUT_S)
        model_output = future.result()
        self.batch_queue.task_done()
        engine_core_outputs = self.scheduler.update_from_output(scheduler_output, model_output)
        return engine_core_outputs


class EngineCoreProc(EngineCore):
    def __init__(self, ...):
        self.input_queue: queue.Queue[Tuple[EngineCoreRequestType, Any]] = queue.Queue()
        self.output_queue: queue.Queue[EngineCoreOutputs] = queue.Queue()

        threading.Thread(target=self.process_input_socket, args=(input_path, ), daemon=True).start()
        threading.Thread(target=self.process_output_socket, args=(output_path, ), daemon=True).start()

        ready_pipe.send({"status": "READY"})

    @staticmethod
    def run_engine_core(*args, **kwargs):
        engine_core = EngineCoreProc(*args, **kwargs)
        engine_core.run_busy_loop()

    def run_busy_loop(self):
        step_fn = (self.step if self.batch_queue is None else self.step_with_batch_queue)

        while True:
            req = self.input_queue.get_nowait()
            self._handle_client_request(*req)

            outputs = step_fn()
            self.output_queue.put_nowait(outputs)

    def _handle_client_request(self, request_type: EngineCoreRequestType, request: Any) -> None:
        if request_type == EngineCoreRequestType.ADD:
            self.add_request(request)
        elif request_type == EngineCoreRequestType.ABORT:
            self.abort_requests(request)

    def process_input_socket(self, input_path: str):
        with zmq_socket_ctx(input_path, zmq.constants.PULL) as socket:
            while True:
                type_frame, data_frame = socket.recv_multipart(copy=False)
                request_type = EngineCoreRequestType(bytes(type_frame.buffer))
                request = decoder.decode(data_frame.buffer)

                self.input_queue.put_nowait((request_type, request))

    def process_output_socket(self, output_path: str):
        with zmq_socket_ctx(output_path, zmq.constants.PUSH) as socket:
            while True:
                outputs = self.output_queue.get()
                encoder.encode_into(outputs, buffer)
                socket.send_multipart((buffer, ), copy=False)
```

`run_engine_core` 调用 run_busy_loop，启动 while 循环, 

1. 从 input_queue 中获取 EngineCoreRequest
2. 调用 _handle_client_request 将新请求放入 scheduler 中调度；
3. 调用 step 处理请求: 调用 scheduler 的 schedule 获取调度信息，调用 executor 的 execute_model 处理请求；
4. 最后将 EngineCoreOutputs 发送到 output_queue 中


```
    def step_with_batch_queue(self) -> Optional[EngineCoreOutputs]:
        """Schedule and execute batches with the batch queue.
        Note that if nothing to output in this step, None is returned.

        The execution flow is as follows:
        1. Try to schedule a new batch if there are unscheduled requests
        and the job queue is not full. If a new batch is scheduled, directly
        return an empty engine core output. In other words, we won't check
        and return model outputs before the batch queue is full.
        2. If there is no new scheduled batch, meaning that the batch queue
        is full or no other requests can be scheduled, we block until the first
        batch in the job queue is finished.
        3. Update the scheduler from the output.
        """
```

    """
    MPClient: base client for multi-proc EngineCore.
        EngineCore runs in a background process busy loop, getting
        new EngineCoreRequests and returning EngineCoreOutputs

        * pushes EngineCoreRequests via input_socket
        * pulls EngineCoreOutputs via output_socket
    
        * AsyncMPClient subclass for AsyncLLM usage
        * SyncMPClient subclass for LLM usage
    """

## Scheduler

```python
# vllm/v1/core/scheduler.py

class Scheduler:

    def __init__(self, scheduler_config, model_config, cache_config, ...):

        self.kv_cache_manager = KVCacheManager()

        self.requests: Dict[str, Request] = {} # req_id -> Request
        self.waiting: Deque[Request] = deque()
        self.running: List[Request] = []
        self.finished_req_ids: Set[str] = set()

    def add_request(self, request: Request) -> None:
        self.waiting.append(request)
        self.requests[request.request_id] = request

    def schedule(self) -> "SchedulerOutput":
        num_scheduled_tokens: Dict[str, int] = {}
        token_budget = self.max_num_scheduled_tokens

        # First, schedule the RUNNING requests.
        req_index = 0
        while req_index < len(self.running) and token_budget > 0:
            request = self.running[req_index]

            num_new_tokens = request.num_tokens_with_spec - request.num_computed_tokens
            while True:
                new_blocks = self.kv_cache_manager.allocate_slots(request, num_new_tokens)
                if new_blocks is None:
                    preempted_req = self.running.pop()
                    self.kv_cache_manager.free(preempted_req)
                    self.waiting.appendleft(preempted_req)
                else:
                    can_schedule = True
                    break
            if not can_schedule:
                break

            scheduled_running_reqs.append(request)
            num_scheduled_tokens[request.request_id] = num_new_tokens
            token_budget -= num_new_tokens
            req_index += 1

            if request.spec_token_ids:
                scheduled_spec_decode_tokens[request.request_id] = (request.spec_token_ids)

        # Next, schedule the WAITING requests.
        if not preempted_reqs:
            while self.waiting and token_budget > 0:
                request = self.waiting[0]

                computed_blocks, num_computed_tokens = self.kv_cache_manager.get_computed_blocks(request)

                num_new_tokens = request.num_tokens - num_computed_tokens

                new_blocks = self.kv_cache_manager.allocate_slots(request, num_new_tokens, computed_blocks)
                if new_blocks is None:
                    break

                self.waiting.popleft()
                self.running.append(request)
                if request.status == RequestStatus.WAITING:
                    scheduled_new_reqs.append(request)
                elif request.status == RequestStatus.PREEMPTED:
                    scheduled_resumed_reqs.append(request)

                num_scheduled_tokens[request.request_id] = num_new_tokens
                token_budget -= num_new_tokens
                request.status = RequestStatus.RUNNING
                request.num_computed_tokens = num_computed_tokens

        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())

        new_reqs_data = [
            NewRequestData.from_request(req,...) for req in scheduled_new_reqs
        ]
        resumed_reqs_data = [ # CachedRequestData
            self._make_cached_request_data(...) for req in scheduled_resumed_reqs
        ]
        running_reqs_data = [ # CachedRequestData
            self._make_cached_request_data(...) for req in scheduled_running_reqs
        ]
        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=new_reqs_data,
            scheduled_cached_reqs=resumed_reqs_data + running_reqs_data,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
        )

        return scheduler_output


    def update_from_output(self, scheduler_output, model_runner_output) -> EngineCoreOutputs:
        sampled_token_ids = model_runner_output.sampled_token_ids

        for request in self.running:
            req_id = request.request_id
            req_index = model_runner_output.req_id_to_index[req_id]
            generated_token_ids = sampled_token_ids[req_index]

            if request.num_computed_tokens >= request.num_tokens:
                for output_token_id in generated_token_ids:
                    request.append_output_token_ids(output_token_id)
                    stopped = self._check_stop(request)
                    if stopped:
                        self._free_request(request)
                        break

            if not stopped:
                new_running.append(request)

        self.running = new_running
        return EngineCoreOutputs(outputs=outputs,...)
```

schedule 返回 `SchedulerOutput` 包含两个列表，

* `scheduled_new_reqs` 来自 waiting queue，即处理新请求
* `scheduled_cached_reqs` 来自 running queue 和 resumed requests，即继续处理正在处理中的请求


new_running: 由 `_check_stop` 实现

* num_tokens_scheduled == 0 : 本次未调度的
* request.num_tokens >= self.max_model_len
* request.num_output_tokens >= request.max_tokens
* last_token_id == request.eos_token_id
* last_token_id in sampling_params.stop_token_ids


Executor 执行的结果通过 `update_from_output` 更新到 `request` 中，
具体而言，从 `model_runner_output.sampled_token_ids` 中取出 `request` 对应的  `output_token_ids`，
通过 `request.append_output_token_ids(output_token_id)` 更新到 `request` 中。


```python
# vllm/v1/request.py 

class Request:
    def __init__(self, request_id, prompt, prompt_token_ids, ...):
        self.prompt = prompt
        self.prompt_token_ids = prompt_token_ids
        self.num_prompt_tokens = len(self.prompt_token_ids)
        self._output_token_ids: List[int] = []
        self._all_token_ids: List[int] = self.prompt_token_ids.copy()
        self.spec_token_ids: List[int] = []
        self.num_computed_tokens = 0

    @property
    def num_tokens(self) -> int:
        return len(self._all_token_ids)

    @property
    def num_tokens_with_spec(self) -> int:
        return len(self._all_token_ids) + len(self.spec_token_ids)

    def append_output_token_ids(self, token_ids) -> None:
        self._output_token_ids.extend(token_ids)
        self._all_token_ids.extend(token_ids)
```

        # NOTE(woosuk) on the scheduling algorithm:
        # There's no "decoding phase" nor "prefill phase" in the scheduler.
        # Each request just has the num_computed_tokens and
        # num_tokens_with_spec. num_tokens_with_spec =
        # len(prompt_token_ids) + len(output_token_ids) + len(spec_token_ids).
        # At each step, the scheduler tries to assign tokens to the requests
        # so that each request's num_computed_tokens can catch up its
        # num_tokens_with_spec. This is general enough to cover
        # chunked prefills, prefix caching, speculative decoding,
        # and the "jump decoding" optimization in the future.

## Executor

用法

```python
# AsyncLLM
executor_class = Executor.get_class(vllm_config)

# EngineCore
self.model_executor = executor_class(vllm_config) 

self.model_executor.initialize(kv_cache_configs)

output = self.model_executor.execute_model(scheduler_output)
```

Excutor 是真正驱动进程执行 GPU 计算的模块。
从 EngineCore 的使用上可以看出，

* 首先是选择具体的 Executor 类实现，进行默认初始化 `__init__`
* 然后是 initialize 初始化 Executor
* 最后是 execute_model 执行模型返回结果


### ExecutorBase

当 ExecutorBase 初始化的时候，会执行 _init_executor，这和 scheduler 中调用的 initialize 不是同一个函数。

* `_init_executor` 根据不同是实现执行 `Worker` 的 `init_worker`, `init_device`, `load_model`.
* `initialize` 会执行 `Worker` 的 `initialize_cache`, `compile_or_warm_up_model`.

`execute_model` 根据不同的实现执行 Worker 的 `execute_model`.

```python
# vllm/executor/executor_base.py

class ExecutorBase(ABC):
    def __init__(self, vllm_config: VllmConfig) -> None:
        self._init_executor()

class DistributedExecutorBase(ExecutorBase):
    def execute_model(self, execute_model_req: ExecuteModelRequest,) -> List[SamplerOutput]:
        if self.parallel_worker_tasks is None:
            self.parallel_worker_tasks = self._run_workers(
                "start_worker_execution_loop",
                async_run_tensor_parallel_workers_only=True)

        driver_outputs = self._driver_execute_model(execute_model_req)
        return driver_outputs

    def collective_rpc(self, ...):
        return self._run_workers(method, *args, **(kwargs or {}))
```

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

    def execute_model(self, scheduler_output,
    ) -> Union[ModelRunnerOutput, Future[ModelRunnerOutput]]:
        output = self.collective_rpc("execute_model", args=(scheduler_output, ))
        return output[0]

```

### RayDistributedExecutor

```python
# vllm/v1/executor/ray_distributed_executor.py

class RayDistributedExecutor(RayDistributedExecutorV0, Executor):
    def execute_model( self, scheduler_output,) -> Union[ModelRunnerOutput, Future[ModelRunnerOutput]]:
        self.forward_dag = self._compiled_ray_dag(enable_asyncio=False)
        refs = self.forward_dag.execute(scheduler_output)
        return refs[0].get()

```

RayDistributedExecutorV0
```python
# vllm/executor/ray_distributed_executor.py

# RayDistributedExecutorV0
class RayDistributedExecutor(DistributedExecutorBase):
    def _init_executor(self) -> None:
        initialize_ray_cluster(self.parallel_config) # ray.init(...)
        self._init_workers_ray(placement_group)

    def _init_workers_ray(self, placement_group, **ray_remote_kwargs):

        bundle_indices = list(map(int, envs.VLLM_RAY_BUNDLE_INDICES.split(",")))
        for rank, bundle_id in enumerate(bundle_indices):
            scheduling_strategy = PlacementGroupSchedulingStrategy(placement_group, bundle_id,)

            worker = ray.remote(num_cpus=0, num_gpus, scheduling_strategy, **ray_remote_kwargs,
                )(RayWorkerWrapper).remote(vllm_config=self.vllm_config, rpc_rank=rank)
            worker_metadata.append(RayWorkerMetaData(worker=worker, created_rank=rank))

        self.workers = [item.worker for item in sorted_worker_metadata] # List[RayWorkerWrapper]

        self._run_workers("adjust_rank", rerank_mapping)
        self._run_workers("update_environment_variables", self._get_env_vars_to_be_updated())

        self._run_workers("init_worker", all_kwargs)
        self._run_workers("init_device")
        self._run_workers("load_model", max_concurrent_workers)

    def execute_model(self, execute_model_req: ExecuteModelRequest) -> List[SamplerOutput]:
        outputs = ray.get(self.forward_dag.execute(execute_model_req))
        return outputs[0]

    def _run_workers(self, method, *args, **kwargs,) -> Any:
        ray_worker_outputs = [worker.execute_method.remote(method, ...) for worker in ray_workers]

        if async_run_tensor_parallel_workers_only:
            return ray_worker_outputs

        ray_worker_outputs = ray.get(ray_worker_outputs)
        return ray_worker_outputs
```

V1 版本的 RayDistributedExecutor 继承 V0 版本的 RayDistributedExecutor，改写了 execute_model 方法，使用 DAG 的实现.

V0 版本的 RayDistributedExecutor 在 `__init__/_init_excutor` 中

1. 启动 ray 集群，然后再集群中启动 RayWorkerWrapper 实现的 worker
2. 在 worker 中执行 `adjust_rank` 和 `update_environment_variables` 方法
3. 在 worker 中执行 `init_worker`, init_device 和 `load_model` 方法

RayDistributedExecutor 中在 worker 中执行的方法通过 `_run_workers` 方法提供，其中支持异步和同步的能力。
在 DistributedExecutorBase 中，`collective_rpc` 即等价于 `_run_workers` 方法。

### MultiprocExecutor

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

    def collective_rpc(self, ...):
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

## Reference

* https://platform.openai.com/docs/api-reference/chat/create

# architecture

## Excutor


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
    def make_worker_process(vllm_config, local_rank: int, rank: int, distributed_init_method: str, input_shm_handle) -> WorkerProcHandle:
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

# dataloader


```python
# python/paddle/io/dataloader/dataloader_iter.py


class _DataLoaderIterBase:
    """
    Iterator implement of DataLoader, will load and feed mini-batch
    data by setting in given dataloader.

    Args:
        loader(instance of DataLoader): instance of `paddle.io.DataLoader`
    """

    def __init__(self, loader):
        self._dataset = loader.dataset
        self._feed_list = loader.feed_list or []
        self._places = loader.places
        self._return_list = loader.return_list
        self._batch_sampler = loader.batch_sampler
        self._drop_last = loader.drop_last
        self._auto_collate_batch = loader.auto_collate_batch
        self._num_workers = loader.num_workers
        self._use_buffer_reader = loader.use_buffer_reader
        self._prefetch_factor = loader.prefetch_factor
        self._use_shared_memory = loader.use_shared_memory
        self._timeout = (
            loader.timeout if loader.timeout > 0 else MP_STATUS_CHECK_INTERVAL
        )
        self._worker_init_fn = loader.worker_init_fn
        self._dataset_kind = loader.dataset_kind
        self._pin_memory = loader.pin_memory
        # LoDTensorBlockingQueue instance for create_py_reader and a thread
        # to put mini-batch data to self._blocking_queue, mini-batch data
        # will be get from:
        # 1. multi-process mode: get data from workers' result queue
        # 2. single-process mode: read mini-batch data in main process
        self._blocking_queue = None
        self._thread = None
        self._thread_done_event = threading.Event()
```

```python
# python/paddle/io/dataloader/dataloader_iter.py

class _DataLoaderIterSingleProcess(_DataLoaderIterBase):
    """
    Single process implement of DataLoaderIter, loading data from
    loader.data in main process
    """

    def __init__(self, loader):
        self._dataset_fetcher = _DatasetKind.create_fetcher(
            self._dataset, ...
        )

        # NOTE: _structrue_infos used to record the data structure of
        # batch to restore batch structure after reading Tensor
        # from blocking_queue in single-process mode. Note that
        # only single process is used in single-process mode, we
        # can record the data structure sequencely in a list without
        # recording the send and recv index
        self._structure_infos = []

        self._blocking_queue_capacity = self._prefetch_factor * len(
            self._places
        )

        self._init_thread()

    def _init_thread(self):
        self._blocking_queue = core.init_lod_tensor_blocking_queue(...)
        self._reader = core.create_py_reader(...)

        self._thread = threading.Thread(
            target=self._thread_loop, args=(_current_expected_place(),)
        )
        self._thread.start()

    def _thread_loop(self, legacy_expected_place):
        while not self._thread_done_event.is_set():
            try:
                indices = next(self._sampler_iter)
                batch = self._dataset_fetcher.fetch(
                    indices, self._thread_done_event
                )

            batch, structure = _flatten_batch(batch)
            try:
                # pack as LoDTensorArray
                array = core.LoDTensorArray()
                for slot in batch:
                    if isinstance(slot, (paddle.Tensor, core.eager.Tensor)):
                        slot = slot.value().get_tensor()
                    elif not isinstance(slot, core.LoDTensor):
                        tmp = core.LoDTensor()
                        tmp.set(slot, core.CPUPlace())
                        slot = tmp

                    array.append(slot)

                try:
                    self._blocking_queue.push(array)

    def __next__(self):
        try:
            if in_dygraph_mode():
                data = core.eager.read_next_tensor_list(
                    self._reader.read_next_list()[0]
                )
                data = _restore_batch(data, self._structure_infos.pop(0))
            else:
                if self._return_list:
                    data = self._reader.read_next_list()
                else:
                    data = self._reader.read_next()
            return data
```

```python
class _DataLoaderIterMultiProcess(_DataLoaderIterBase):
    def __init__(self, loader):
        super().__init__(loader)

        self._data_queue = None

        self._outstanding_capacity = self._prefetch_factor * max(
            self._num_workers, len(self._places)
        )

        # see _try_put_indices
        self._thread_lock = threading.Lock()
        self._init_workers()
        for _ in range(self._outstanding_capacity):
            self._try_put_indices()

        self._init_thread()

    def _init_workers(self):
        self._data_queue = multiprocessing.Queue()

        for i in range(self._num_workers):
            indices_queue = multiprocessing.Queue()
            self._indices_queues.append(indices_queue)
            worker = multiprocessing.Process( target=_worker_loop, args=(self._dataset, ...),)
            worker.start()
            self._workers.append(worker)

    def _init_thread(self):
        self._blocking_queue = core.init_lod_tensor_blocking_queue(
            core.Variable(), self._outstanding_capacity, len(self._places) > 1
        )
        self._reader = core.create_py_reader(self._blocking_queue, ...)

        self._thread = threading.Thread(
            target=self._thread_loop, args=(_current_expected_place(),)
        )
        self._thread.start()

    def _reset(self):
        # resume iteration in following steps
        # 1. Resume workers, clear worker caches
        # put _ResumeIteration to all worker as resume iteration flag
        with self._thread_lock:
            self._resume_worker_cnt = self._num_workers
            for worker_id in range(self._num_workers):
                self._indices_queues[worker_id].put(_ResumeIteration())
                self._batches_outstanding += 1
        # all flag will be check in _thread_loop, simply wait here
        while self._resume_worker_cnt > 0:
            time.sleep(0.5)

        # 2. clear blocking_queue caches
        # in order not to restart the thread, we just clear
        # the blocking_queue cachees instead of recreating one
        while self._blocking_queue.size() >= len(self._places):
            if in_dygraph_mode():
                data = core.eager.read_next_tensor_list(
                    self._reader.read_next_list()[0]
                )
            else:
                if self._return_list:
                    self._reader.read_next_list()
                else:
                    data = self._reader.read_next()

        # 3. reset all states
        self._send_idx = 0
        self._rcvd_idx = 0
        self._batches_outstanding = 0
        self._task_infos = {}
        self._structure_infos = []

        # set all worker status available
        self._worker_status = [True] * self._num_workers

        # 4. reset _sampler_iter and put prefetch indices to start next epoch
        # init workers and indices queues and put 2 indices in each indices queue
        self._sampler_iter = iter(self._index_sampler)
        for _ in range(self._outstanding_capacity):
            self._try_put_indices()

    def _thread_loop(self, legacy_expected_place):
        while not self._thread_done_event.is_set():
            batch = self._get_data()
            if not self._thread_done_event.is_set():
                    try:
                        # pack as LoDTensorArray
                        array = core.LoDTensorArray()
                        if self._use_shared_memory:
                            for tensor in batch:
                                array.append(tensor)
                        else:
                            # LoDTensor not in shared memory is not
                            # serializable, cannot be create in workers
                            for slot in batch:
                                if isinstance(
                                    slot, (paddle.Tensor, core.eager.Tensor)
                                ):
                                    slot = slot.value().get_tensor()
                                elif not isinstance(slot, core.LoDTensor):
                                    tmp = core.LoDTensor()
                                    tmp.set(slot, core.CPUPlace())
                                    slot = tmp
                                array.append(slot)


                        if not self._blocking_queue.push(array):
                            self._blocking_queue.close()

    def _get_data(self):
        while not self._thread_done_event.is_set():
            # For IterableDataset, batch indices is generated infinitely
            # for each worker to raise StopIteration, but a StopIteration
            # raising process will discard a batch indices which is count
            # in _send_idx but will not increase _rcvd_idx, so we check
            # whether the worker is still alive here to skip the discarded
            # batch indices and increase _rcvd_idx
            if self._dataset_kind == _DatasetKind.ITER:
                while self._rcvd_idx < self._send_idx:
                    info = self._task_infos[self._rcvd_idx]
                    if len(info) == 3 or self._worker_status[info[0]]:
                        break
                    del self._task_infos[self._rcvd_idx]
                    self._rcvd_idx += 1
                    self._batches_outstanding -= 1
                else:
                    # NOTE: when _rcvd_idx catch up _send_idx, which means
                    #       one of following:
                    #       1. all 2 * num_workers batches have been loaded
                    #          and stored in _blocking_queue
                    #       2. all data drained
                    #       we need to let _thread blocking at _data_queue
                    #       get_data to inoccupy CPU, otherwise may occupy
                    #       CPU time for model running
                    # NOTE: in persistent workers mode, do not check data
                    #       drained here, simply let it go to _data_queue
                    #       reading to get _ResumeIteration
                    if not self._persistent_workers:
                        # NOTE: _rcvd_idx and _send_idx only record batches among
                        #       workers, if batches among workers drained, there
                        #       may also be data in blocking queue
                        if self._batches_outstanding < len(self._places):
                            return None

            if (
                self._rcvd_idx in self._task_infos
                and len(self._task_infos[self._rcvd_idx]) == 3
            ):
                info = self._task_infos.pop(self._rcvd_idx)
                self._structure_infos.append(info[2])
                return info[1]

            try:
                # [ avoid hang ]: main process may blocking at _reader.read_next when
                # KeyboardInterrupt, we do following tradeoff:
                # 1. get data with timeout, MP_STATUS_CHECK_INTERVAL(5s) as timeout
                #    default, if KeyboardInterrupt blocking, failed workers will be
                #    checked and raise RuntimeError to quit DataLoader in timeout
                #    exception handling.
                # 2. if get data timeout and check workers all alive, continue to
                #    get data again
                data = self._data_queue.get(timeout=self._timeout)
            except Exception as e:
                # check if thread done event set when waiting data
                if self._thread_done_event.is_set():
                    continue

                # check failed workers
                failed_workers = []
                for i, w in enumerate(self._workers):
                    if self._worker_status[i] and not w.is_alive():
                        failed_workers.append(w)
                        self._shutdown_worker(i)
                if len(failed_workers) > 0:
                    self._exit_thread_unexpectedly()
                    pids = ', '.join(str(w.pid) for w in failed_workers)
                    raise RuntimeError(
                        "DataLoader {} workers exit unexpectedly, "
                        "pids: {}".format(len(failed_workers), pids)
                    )

                # get(timeout) will call _poll(timeout) and may raise IOError
                if isinstance(e, queue.Empty) or isinstance(e, IOError):
                    # continue on timeout to keep getting data from queue
                    continue

                self._exit_thread_unexpectedly()
                logging.error(
                    "DataLoader reader thread failed({}) to read data from "
                    "workers' result queue.".format(e)
                )
                raise e
            else:
                if self._dataset_kind == _DatasetKind.ITER and isinstance(
                    data, _IterableDatasetStopIteration
                ):
                    # if a worker get StopIteraion, we shutdown this worker,
                    # note that this batch indices to trigger StopIteration
                    # is discard, outstanding batch number should be decrease
                    # and another indices should be put for other workers
                    # may still working.
                    if self._persistent_workers:
                        self._worker_status[data.worker_id] = False
                    else:
                        self._shutdown_worker(data.worker_id)
                        self._batches_outstanding -= 1
                    self._try_put_indices()
                    continue

                idx, batch, structure = data

                if (
                    isinstance(idx, _ResumeIteration)
                    and batch is None
                    and structure is None
                ):
                    return idx

                if isinstance(batch, _WorkerException):
                    self._exit_thread_unexpectedly()
                    batch.reraise()

                if idx == self._rcvd_idx:
                    del self._task_infos[idx]
                    self._structure_infos.append(structure)
                    return batch
                else:
                    self._task_infos[idx] += (batch, structure)
                    continue

    def _try_put_indices(self):
        assert (
            self._batches_outstanding <= self._outstanding_capacity
        ), "too many indices have been put to queue"
        # In multi-process mode for IterableDataset, _try_put_indices will
        # be called both in main process(for our implement has blocking queue,
        # and blocking queue read is in main process) and thread, which may
        # cause error following error
        #   1. "ValueError: generator already executing" in next(self._sampler_iter)
        #   2. re-enter in increase _send_idx
        # add a lock for threading save, for _try_put_indices is only a slight
        # function which is not in data reading pipeline, this lock almost no
        # influence on performance
        with self._thread_lock:
            try:
                indices = next(self._sampler_iter)
            except StopIteration:
                return

            for i in range(self._num_workers):
                worker_idx = next(self._workers_idx_cycle)
                if self._worker_status[worker_idx]:
                    break
            else:
                return

            self._indices_queues[worker_idx].put((self._send_idx, indices))
            self._task_infos[self._send_idx] = (worker_idx,)
            self._batches_outstanding += 1
            self._send_idx += 1

    def __del__(self):
        self._try_shutdown_all()

    def _shutdown_on_exit(self):
        self._try_shutdown_all(1)

    def __next__(self):
        if in_profiler_mode():
            trace_event = profiler.RecordEvent(
                name="_DataLoaderIterMultiProcess",
                event_type=profiler.TracerEventType.Dataloader,
            )
            trace_event.begin()
        try:
            benchmark().check_if_need_record(self)
            benchmark().before_reader()
            # _batches_outstanding here record the total batch data number
            # in 'from after _try_put_indices to beforeoutput data', this
            # value should be _outstanding_capacity if data is not drained,
            # if _batches_outstanding is less than _places number, there are
            # no enough data to generate next output, close blocking_queue and
            # set _thread_done_event here, py_reader will raise StopIteration,
            # end workers and indices_queues in StopIteration handling
            if self._batches_outstanding < len(self._places):
                if self._persistent_workers:
                    raise StopIteration
                else:
                    self._thread_done_event.set()
                    self._blocking_queue.close()

            if in_dygraph_mode():
                data = core.eager.read_next_tensor_list(
                    self._reader.read_next_list()[0]
                )
                data = _restore_batch(data, self._structure_infos.pop(0))
            else:
                if self._return_list:
                    data = self._reader.read_next_list()
                    for i in range(len(data)):
                        data[i] = data[i]._move_to_list()
                    structs = [
                        self._structure_infos.pop(0)
                        for _ in range(len(self._places))
                    ]
                    data = [_restore_batch(d, s) for d, s in zip(data, structs)]
                    # static graph organized data on multi-device with list, if
                    # place number is 1, there is only 1 device, extra the data
                    # from list for devices to be compatible with dygraph mode
                    if len(self._places) == 1:
                        data = data[0]
                else:
                    data = self._reader.read_next()
            self._on_output_batch()
            benchmark().after_reader()
            return data
        except StopIteration:
            if not self._persistent_workers:
                self._reader.shutdown()
                self._try_shutdown_all()
            raise
        finally:
            if in_profiler_mode():
                trace_event.end()

    def _on_output_batch(self):
        for _ in range(len(self._places)):
            self._batches_outstanding -= 1
            self._try_put_indices()
```

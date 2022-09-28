# nccl

## Example

one device per process/thread 为常用模式

1. ncclGetUniqueId 生成 ID,

NCCL 建立通信域需要依赖共同的 nccl ID, 通常在 0 号节点上调用 ncclGetUniqueId API 生成 ID，
然后 broadcast 到所有节点上，这里 broadcast 的媒介 NCCL 并未提供方法，也没有限制。
可以通过 mpi、gloo、tcp 等方式在节点间同步。

```cpp
ncclUniqueId id;
if (myRank == 0) ncclGetUniqueId(&id);
```

2. ncclCommInitRank 建立通信域 

NCCL 使用 (id, rank, size) 三元组建立通信域。

```cpp
ncclComm_t comm;
ncclCommInitRank(&comm, nRanks, id, myRank);
```

3. ncclAllReduce 使用 NCCL 通信

通过建立好的通信域，可以调用 NCCL 提供的 API 进行集合通信。

```cpp
ncclAllReduce( ... , comm);
```

4. ncclCommDestroy 销毁通信域

使用 ncclCommDestroy API 销毁通信域，释放资源，另 ncclCommAbort 可用于标记释放通信域，实现容错等流程。

```cpp
ncclCommDestroy(comm);
```

## API

重点使用的 API, 可以说是使用 NCCL 的目的就是使用这些 API 进行通信交换数据。

* Collective Communication Functions
    * ncclAllReduce
    * ncclBroadcast
    * ncclReduce
    * ncclAllGather
    * ncclReduceScatter

为了实现上述操作需要建立基本的连接等等操作，重点围绕通信域概念。

* Communicator Creation and Management Functions
    * ncclGetVersion
    * ncclGetUniqueId
    * ncclCommInitRank
    * ncclCommInitAll
    * ncclCommInitRankConfig
    * ncclCommDestroy
    * ncclCommAbort
    * ncclCommGetAsyncError
    * ncclCommCount
    * ncclCommCuDevice
    * ncclCommUserRank

为了讲述，进行如下定义，才有前述的操作。

* Types
    * ncclComm_t
    * ncclResult_t
    * ncclDataType_t
    * ncclRedOp_t
    * ncclConfig_t

为了高效和优化引入逻辑上的组概念。

* Group Calls
    * ncclGroupStart
    * ncclGroupEnd

引入点对点通信，可以实现比如 all-to-all 操作。

* Point To Point Communication Functions
    * ncclSend
    * ncclRecv

## Init

```cpp
// nccl/src/init.cc

ncclResult_t ncclGetUniqueId(ncclUniqueId* out) {
  NCCLCHECK(ncclInit());
  return bootstrapGetUniqueId(out);
}

static ncclResult_t ncclInit() {
  if (initialized) return ncclSuccess;
  pthread_mutex_lock(&initLock);
  if (!initialized) {
    initEnv();
    initGdrCopy();
    maxLocalSizeBytes = ncclKernMaxLocalSize();
    NCCLCHECK(initNet());
    INFO(NCCL_INIT, "Using network %s", ncclNetName());
    initialized = true;
  }
  pthread_mutex_unlock(&initLock);
  return ncclSuccess;
}
```

```cpp
// nccl/src/bootstrap.cc  

ncclResult_t bootstrapGetUniqueId(ncclUniqueId* id) {
  static_assert(sizeof(union socketAddress) < sizeof(ncclUniqueId), "NetId does not fit inside ncclUniqueId");
  memset(id, 0, sizeof(ncclUniqueId));
  union socketAddress* connectAddr = (union socketAddress*) id;

  char* env = getenv("NCCL_COMM_ID");
  if (env) {
    INFO(NCCL_ENV, "NCCL_COMM_ID set by environment to %s", env);
    if (GetSocketAddrFromString(connectAddr, env) != ncclSuccess) {
      WARN("Invalid NCCL_COMM_ID, please use format: <ipv4>:<port> or [<ipv6>]:<port> or <hostname>:<port>");
      return ncclInvalidArgument;
    }
  } else {
    memcpy(id, &bootstrapNetIfAddr, sizeof(union socketAddress));
    NCCLCHECK(bootstrapCreateRoot(id, false));
  }

  return ncclSuccess;
}

```

## Cuda Graph

```cpp
cudaGraph_t graph;
cudaStreamBeginCapture(stream);
kernel_A<<< ..., stream >>>(...);
kernel_B<<< ..., stream >>>(...);
ncclAllreduce(..., stream);
kernel_C<<< ..., stream >>>(...);
cudaStreamEndCapture(stream, &graph);

cudaGraphExec_t instance;
cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
cudaGraphLaunch(instance, stream);
cudaStreamSynchronize(stream);
```

```cpp
bool graphCreated=false;
cudaGraph_t graph;
cudaGraphExec_t instance;
for(int istep=0; istep<NSTEP; istep++){
  if(!graphCreated){
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    for(int ikrnl=0; ikrnl<NKERNEL; ikrnl++){
      shortKernel<<<blocks, threads, 0, stream>>>(out_d, in_d);
    }
    cudaStreamEndCapture(stream, &graph);
    cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
    graphCreated=true;
  }
  cudaGraphLaunch(instance, stream);
  cudaStreamSynchronize(stream);
}
```

## References

* [examples](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/examples.html)

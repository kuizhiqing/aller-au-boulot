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

## Workflow

### ncclGetUniqueId

创建 `ncclGetUniqueId` 时会首先调用初始化函数 `ncclInit` 确认网络已经初始化, 然后调用 `bootstrapGetUniqueId` 创建 `ncclUniqueId`。

```cpp
// nccl/src/init.cc

ncclResult_t ncclGetUniqueId(ncclUniqueId* out) {
  NCCLCHECK(ncclInit());
  return bootstrapGetUniqueId(out);
}
```

`ncclInit` 分为两部分

* `bootstrapNetInit` 调用 `ncclFindInterfaces` 完成获取 Interface 信息，赋值 `ncclSocketAddress bootstrapNetIfAddr`
* `ncclNetPluginInit` 加载 net plugin

```cpp
// nccl/src/init.cc

static ncclResult_t ncclInit() {
  NCCLCHECK(bootstrapNetInit());
  NCCLCHECK(ncclNetPluginInit());
  return ncclSuccess;
}
```

从定义

```cpp
#define NCCL_UNIQUE_ID_BYTES 128
typedef struct { char internal[NCCL_UNIQUE_ID_BYTES]; } ncclUniqueId;
```

可以看出 `ncclUniqueId` 就是 128 个 char 构成的 struct。


```cpp
// nccl/src/bootstrap.cc  

static union socketAddress bootstrapNetIfAddr;

ncclResult_t bootstrapGetUniqueId(ncclUniqueId* id) {
  memset(id, 0, sizeof(ncclUniqueId));

  char* env = getenv("NCCL_COMM_ID");
  if (env) {
    if (ncclGetSocketAddrFromString(connectAddr, env) != ncclSuccess) {
      return ncclInvalidArgument;
    }
  } else {
    memcpy(id, &bootstrapNetIfAddr, sizeof(union ncclSocketAddress));
    NCCLCHECK(bootstrapCreateRoot(id, false));
  }

  return ncclSuccess;
}
```

创建主节点监听线程，`ncclUniqueId` 的本质是 `socketAddress`，

```cpp
ncclResult_t bootstrapCreateRoot(ncclUniqueId* id, bool idFromEnv) {
  union socketAddress* connectAddr = (union socketAddress*) id;
  int listenFd;
  NCCLCHECK(createListenSocket(&listenFd, connectAddr));
  pthread_t thread;
  pthread_create(&thread, NULL, bootstrapRoot, (void*)(uint64_t)listenFd);
  return ncclSuccess;
}
```

> 使用 `NCCL_COMM_ID` 时，该函数在 `ncclCommInitRankDev` 里调用启动 root 上的 tcp 监听服务。

### ncclCommInitRank

`ncclCommInitRank` 初始化调用 `ncclCommInitRankDev` 完成初始化，包括

* 如果使用 `NCCL_COMM_ID` 且 rank 为 0 则需要启动 root 服务；
* 确保已经 `ncclInit`
* 调用 `ncclCommInitRankFunc` 初始化，主要调用 `initTransportsRank` 完成 `rendezvous`.

```cpp
// nccl/src/init.cc

ncclResult_t ncclCommInitRank(ncclComm_t* newcomm, int nranks, ncclUniqueId commId, int myrank) {
  NCCLCHECK(ncclCommInitRankDev(newcomm, nranks, commId, myrank, cudaDev, NULL));
  return ncclSuccess;
}

static ncclResult_t ncclCommInitRankDev(ncclComm_t* newcomm, int nranks, ncclUniqueId commId, int myrank, int cudaDev, ncclConfig_t *config) {
  char* env = getenv("NCCL_COMM_ID");
  if (env && myrank == 0) {
    NCCLCHECKGOTO(bootstrapCreateRoot(&commId, true), res, fail);
  }

  NCCLCHECKGOTO(ncclInit(), res, fail);

  struct ncclCommInitRankAsyncJob *job = NULL;
  NCCLCHECKGOTO(ncclAsyncLaunch(&job->base, ncclCommInitRankFunc, NULL, free, comm), res, fail);

  return ncclGroupErrCheck(res);
}

static ncclResult_t ncclCommInitRankFunc(struct ncclAsyncJob* job_) {
  NCCLCHECKGOTO(initTransportsRank(*newcomm, &commId), res, cleanup);

  comm->initState = ncclSuccess;
  return ncclSuccess;
}
```

`initTransportsRank` 是初始化中最为复杂的部分，主要包括

* 通过 bootstrapAllGather 把所有 peer 的信息收集在一起
* 计算 3 个 ncclTopoGraph: ring/tree/colnet 
* 建立 p2p/ring/tree 等链接

```cpp

static ncclResult_t initTransportsRank(struct ncclComm* comm, ncclUniqueId* commId) {
  // We use 2 AllGathers
  // 1. { peerInfo, comm, compCap}
  // 2. { nChannels, graphInfo, topoRanks }

  NCCLCHECK(bootstrapInit(commId, comm));

  NCCLCHECK(bootstrapAllGather(comm->bootstrap, comm->peerInfo, sizeof(struct ncclPeerInfo)));

  // Topo detection / System graph creation
  NCCLCHECK(ncclTopoGetSystem(comm, &comm->topo));
  // Compute paths between GPUs and NICs
  NCCLCHECK(ncclTopoComputePaths(comm->topo, comm));
  // Remove inaccessible GPUs and unused NICs
  NCCLCHECK(ncclTopoTrimSystem(comm->topo, comm));
  // Recompute paths after trimming
  NCCLCHECK(ncclTopoComputePaths(comm->topo, comm));
  // Init search
  NCCLCHECK(ncclTopoSearchInit(comm->topo));
  // Print final topology
  NCCLCHECK(ncclTopoPrint(comm->topo));

  // Set Affinity to a CPU local the our GPU, so that all memory we allocate
  // on the host is local.
  NCCLCHECK(ncclTopoGetCpuAffinity(comm->topo, comm->rank, &comm->cpuAffinity));

  // Launch proxy service thread
  NCCLCHECK(ncclProxyCreate(comm));

  // Get rings and trees
  struct ncclTopoGraph ringGraph;
  NCCLCHECK(ncclTopoCompute(comm->topo, &ringGraph));
  NCCLCHECK(ncclTopoPrintGraph(comm->topo, &ringGraph));

  struct ncclTopoGraph treeGraph;
  NCCLCHECK(ncclTopoCompute(comm->topo, &treeGraph));
  NCCLCHECK(ncclTopoPrintGraph(comm->topo, &treeGraph));

  struct ncclTopoGraph collNetGraph;
  NCCLCHECK(ncclTopoCompute(comm->topo, &collNetGraph));
  NCCLCHECK(ncclTopoPrintGraph(comm->topo, &collNetGraph));

  // Determine local CollNet support before all-gather

  // AllGather3 - begin
  struct ncclGraphInfo {
    int pattern;
    int nChannels;
    int sameChannels;
    float bwIntra;
    float bwInter;
    int typeIntra;
    int typeInter;
  };

  struct {
    int netDev;
    int collNetSupport;
    struct ncclGraphInfo tree;
    struct ncclGraphInfo ring;
    struct ncclGraphInfo collNet;
    struct ncclTopoRanks topoRanks;
  } *allGather3Data;

  NCCLCHECK(bootstrapAllGather(comm->bootstrap, allGather3Data, sizeof(*allGather3Data)));

  // Connect with prev/next for each ring
  for (int c=0; c<comm->nChannels; c++) {
    NCCLCHECKGOTO(setupChannel(comm, c, rank, nranks, rings+c*nranks), ret, affinity_restore);
    NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, 1, &channel->ring.prev, 1, &channel->ring.next, 0), ret, affinity_restore);
  }
  NCCLCHECKGOTO(ncclTransportP2pSetup(comm, &ringGraph, 0), ret, affinity_restore);

  // Connect Trees
  for (int c=0; c<comm->nChannels; c++) {
    NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, NCCL_MAX_TREE_ARITY, channel->tree.down, 1, &channel->tree.up, 0), ret, affinity_restore);
    NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, 1, &channel->tree.up, NCCL_MAX_TREE_ARITY, channel->tree.down, 0), ret, affinity_restore);
  }
  NCCLCHECKGOTO(ncclTransportP2pSetup(comm, &treeGraph, 0), ret, affinity_restore);

  // Compute nChannels per peer for p2p
  NCCLCHECK(ncclTopoComputeP2pChannels(comm));

  /* Local intra-node barrier */
  NCCLCHECK(bootstrapBarrier(comm->bootstrap, comm->localRankToRank, comm->localRank, comm->localRanks, comm->localRankToRank[0]));

  return ncclSuccess;
}
```

其中 `bootstrapInit` 和 `bootstrapAllGather` 完成建立 socket 和连接的操作。

```cpp
// src/bootstrap.cc

ncclResult_t bootstrapInit(ncclUniqueId * id, struct ncclComm* comm) {
  // Create socket for other ranks to contact me
  NCCLCHECK(ncclSocketListen(&state->listenSock));
  memcpy(&info.extAddressListen, &state->listenSock.addr, sizeof(union ncclSocketAddress));

  // Create socket for root to contact me
  NCCLCHECK(ncclSocketListen(&listenSockRoot));
  memcpy(&info.extAddressListenRoot, &listenSockRoot.addr, sizeof(union ncclSocketAddress));

  // send info on my listening socket to root
  NCCLCHECK(ncclSocketConnect(&sock));
  NCCLCHECK(bootstrapNetSend(&sock, &info, sizeof(info)));

  // get info on my "next" rank in the bootstrap ring from root
  NCCLCHECK(ncclSocketAccept(&sock, &listenSockRoot));
  NCCLCHECK(bootstrapNetRecv(&sock, &state->ringSendSocket.addr, sizeof(union ncclSocketAddress)));

  NCCLCHECK(ncclSocketConnect(&state->ringSendSocket));
  // Accept the connect request from the previous rank in the AllGather ring
  NCCLCHECK(ncclSocketAccept(&state->ringRecvSocket, &state->listenSock));

  // AllGather all listen handlers
  NCCLCHECK(ncclCalloc(&state->peerCommAddresses, nranks));
  memcpy(state->peerCommAddresses+rank, &state->listenSock.addr, sizeof(union ncclSocketAddress));
  NCCLCHECK(bootstrapAllGather(state, state->peerCommAddresses, sizeof(union ncclSocketAddress)));

  // Create the service proxy
  NCCLCHECK(ncclCalloc(&state->peerProxyAddresses, nranks));
  struct ncclSocket* proxySocket;
  NCCLCHECK(ncclCalloc(&proxySocket, 1));
  NCCLCHECK(ncclSocketInit(proxySocket, &bootstrapNetIfAddr, NULL, 0));
  NCCLCHECK(ncclSocketListen(proxySocket));
  memcpy(state->peerProxyAddresses+rank, &proxySocket->addr, sizeof(union ncclSocketAddress));
  NCCLCHECK(bootstrapAllGather(state, state->peerProxyAddresses, sizeof(union ncclSocketAddress)));
  NCCLCHECK(ncclProxyInit(comm, proxySocket, state->peerProxyAddresses));

  return ncclSuccess;
}

ncclResult_t bootstrapAllGather(void* commState, void* allData, int size) {
  /* Simple ring based AllGather
   * At each step i receive data from (rank-i-1) from left
   * and send previous step's data from (rank-i) to right
   */
  for (int i=0; i<nranks-1; i++) {
    size_t rslice = (rank - i - 1 + nranks) % nranks;
    size_t sslice = (rank - i + nranks) % nranks;

    // Send slice to the right
    NCCLCHECK(bootstrapNetSend(&state->ringSendSocket, data+sslice*size, size));
    // Recv slice from the left
    NCCLCHECK(bootstrapNetRecv(&state->ringRecvSocket, data+rslice*size, size));
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

## ENV

* `NCCL_GRAPH_DUMP_FILE` 可以保存网络拓扑信息到文件

## References

* [examples](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/examples.html)

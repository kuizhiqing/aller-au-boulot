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

## NCCL init

```cpp
#include <stdio.h>
#include "cuda_runtime.h"
#include "nccl.h"
#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>

#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",       \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",       \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

static uint64_t getHostHash(const char *string) {
    // Based on DJB2, result = result * 33 + char
    uint64_t result = 5381;
    for (int c = 0; string[c] != '\0'; c++) {
        result = ((result << 5) + result) + string[c];
    }
    return result;
}

static void getHostName(char *hostname, int maxlen) {
    gethostname(hostname, maxlen);
    for (int i = 0; i < maxlen; i++) {
        if (hostname[i] == '.') {
            hostname[i] = '\0';
            return;
        }
    }
}

int main(int argc, char *argv[]) {

    int bsize = 32 * 1024 * 1024;

    const int rank = atoi(getenv("RANK"));
    const int size = atoi(getenv("SIZE"));

    ncclUniqueId id;
    ncclComm_t comm;
    float *sendbuff, *recvbuff;
    cudaStream_t s;

    ncclGetUniqueId(&id);

    // picking a GPU based on localRank, allocate device buffers
    CUDACHECK(cudaSetDevice(rank));
    CUDACHECK(cudaMalloc(&sendbuff, bsize * sizeof(float)));
    CUDACHECK(cudaMalloc(&recvbuff, bsize * sizeof(float)));
    CUDACHECK(cudaStreamCreate(&s));

    // initializing NCCL
    NCCLCHECK(ncclCommInitRank(&comm, size, id, rank));

    // communicating using NCCL
    NCCLCHECK(ncclAllReduce((const void *) sendbuff, (void *) recvbuff,
                            bsize, ncclFloat, ncclSum,
                            comm, s));

    // completing NCCL operation by synchronizing on the CUDA stream
    CUDACHECK(cudaStreamSynchronize(s));

    // free device buffers
    CUDACHECK(cudaFree(sendbuff));
    CUDACHECK(cudaFree(recvbuff));

    // finalizing NCCL
    ncclCommDestroy(comm);

    printf("[Rank %d] Success \n", rank);
    return 0;
}
```

```shell
NCCL INFO Bootstrap : Using eth0:10.10.1.1<0>
NCCL INFO NET/Plugin : Plugin load returned 17 : libnccl-net.so: cannot open shared object file: No such file or directory.
NCCL INFO cudaDriverVersion 11020
NCCL version 2.14.3+cuda11.2
56.247345 bootstrapRoot:103 NCCL TRACE BEGIN
NCCL INFO Failed to open libibverbs.so[.1]
NCCL INFO NET/Socket : Using [0]eth0:10.10.1.1<0>
NCCL INFO Using network Socket
62.445105 commAlloc:333 NCCL TRACE comm 0x212a200 rank 0 nranks 2 cudaDev 0 busId 3f000
62.464643 initTransportsRank:515 NCCL TRACE comm 0x212a200, commHash 17da27e246bdeae5, rank 0 nranks 2 - BEGIN
62.472759 bootstrapInit:225 NCCL TRACE rank 0 nranks 2
64.823384 bootstrapRoot:134 NCCL TRACE Received connect from rank 0 total 1/2
713528.501653 bootstrapRoot:134 NCCL TRACE Received connect from rank 1 total 2/2
713528.526803 bootstrapRoot:136 NCCL TRACE COLLECTED ALL 2 HANDLES
713533.526805 bootstrapRoot:149 NCCL TRACE SENT OUT ALL 2 HANDLES
713533.563041 bootstrapRoot:158 NCCL TRACE DONE
713534.258114 bootstrapAllGather:296 NCCL TRACE rank 0 nranks 2 size 28
713536.434154 bootstrapAllGather:312 NCCL TRACE rank 0 nranks 2 size 28 - DONE
713536.462180 bootstrapAllGather:296 NCCL TRACE rank 0 nranks 2 size 28
713536.483120 bootstrapAllGather:312 NCCL TRACE rank 0 nranks 2 size 28 - DONE
713536.493973 bootstrapInit:285 NCCL TRACE rank 0 nranks 2 - DONE
713536.544262 getHostHash:112 NCCL TRACE unique hostname 'xxx.xxx.xxx.xxx'
713536.568319 getPidHash:132 NCCL TRACE unique PID '36548pid:[4026534279]'
713563.230542 bootstrapAllGather:296 NCCL TRACE rank 0 nranks 2 size 64
713563.325008 bootstrapAllGather:312 NCCL TRACE rank 0 nranks 2 size 64 - DONE
715298.136366 ncclTopoGetCpuAffinity:754 NCCL TRACE Current affinity for GPU 0 is ff,ffffffff
715298.149614 ncclTopoGetCpuAffinity:765 NCCL TRACE CPU GPU affinity for GPU 0 is 0fffff
NCCL INFO Setting affinity for GPU 0 to 0fffff
715305.454682 bootstrapAllGather:296 NCCL TRACE rank 0 nranks 2 size 988
715305.544090 bootstrapAllGather:312 NCCL TRACE rank 0 nranks 2 size 988 - DONE
715305.552080 initTransportsRank:698 NCCL TRACE hostHash[0] 69ce2bece884b496 localRank 0 localRanks 2 localRank0 0
NCCL INFO Channel 00/04 :    0   1
NCCL INFO Channel 01/04 :    0   1
NCCL INFO Channel 02/04 :    0   1
NCCL INFO Channel 03/04 :    0   1
715305.591060 initTransportsRank:768 NCCL TRACE rank 0 nranks 2 - BUILT 4 TREES/RINGS
NCCL INFO Trees [0] 1/-1/-1->0->-1 [1] 1/-1/-1->0->-1 [2] 1/-1/-1->0->-1 [3] 1/-1/-1->0->-1
715305.603265 setupChannel:458 NCCL TRACE rank 0 nranks 2
715334.994056 ncclTransportP2pConnect:43 NCCL TRACE nsend 1 nrecv 1
715335.016927 setupChannel:458 NCCL TRACE rank 0 nranks 2
715335.038300 ncclTransportP2pConnect:43 NCCL TRACE nsend 1 nrecv 1
715335.042547 setupChannel:458 NCCL TRACE rank 0 nranks 2
715335.059655 ncclTransportP2pConnect:43 NCCL TRACE nsend 1 nrecv 1
715335.062755 setupChannel:458 NCCL TRACE rank 0 nranks 2
715335.078080 ncclTransportP2pConnect:43 NCCL TRACE nsend 1 nrecv 1
NCCL INFO Channel 00/0 : 0[3f000] -> 1[40000] via P2P/IPC
NCCL INFO Channel 01/0 : 0[3f000] -> 1[40000] via P2P/IPC
NCCL INFO Channel 02/0 : 0[3f000] -> 1[40000] via P2P/IPC
NCCL INFO Channel 03/0 : 0[3f000] -> 1[40000] via P2P/IPC
NCCL INFO Connected all rings
715759.057432 ncclTransportP2pConnect:43 NCCL TRACE nsend 1 nrecv 3
715759.070377 ncclTransportP2pConnect:43 NCCL TRACE nsend 3 nrecv 1
715759.078084 ncclTransportP2pConnect:43 NCCL TRACE nsend 1 nrecv 3
715759.085182 ncclTransportP2pConnect:43 NCCL TRACE nsend 3 nrecv 1
715759.095093 ncclTransportP2pConnect:43 NCCL TRACE nsend 1 nrecv 3
715759.104299 ncclTransportP2pConnect:43 NCCL TRACE nsend 3 nrecv 1
715759.113734 ncclTransportP2pConnect:43 NCCL TRACE nsend 1 nrecv 3
715759.122855 ncclTransportP2pConnect:43 NCCL TRACE nsend 3 nrecv 1
NCCL INFO Connected all trees
715760.166631 initTransportsRank:885 NCCL TRACE rank 0 nranks 2 - CONNECTED 4 RINGS AND TREES
NCCL INFO threadThresholds 8/8/64 | 16/8/64 | 512 | 512
NCCL INFO 4 coll channels, 4 p2p channels, 4 p2p channels per peer
715821.176226 initTransportsRank:1006 NCCL TRACE pidHash[0] 6dfbb6f42b423cdb intraProcRank 0 intraProcRanks 1 intraProcRank0 0
715890.912294 bootstrapBarrier:331 NCCL TRACE rank 0 nranks 2 tag 0 - ENTER
715895.718005 bootstrapBarrier:346 NCCL TRACE rank 0 nranks 2 tag 0 - DONE
715895.838650 initTransportsRank:1054 NCCL TRACE rank 0 nranks 2 - DONE
NCCL INFO comm 0x212a200 rank 0 nranks 2 cudaDev 0 busId 3f000 - Init COMPLETE
715939.466883 ncclCommDestroy:1476 NCCL TRACE comm 0x212a200 rank 0 nRanks 2 cudaDev 0 busId 3f000
715939.492722 commReclaim:1408 NCCL TRACE commReclaim: reclaim comm 0x212a200 rank 0 state 0
715939.500821 commDestroySync:1309 NCCL TRACE Destroying comm 0x212a200 rank 0 abortFlag 0 asyncResult 0
NCCL INFO comm 0x212a200 rank 0 nranks 2 cudaDev 0 busId 3f000 - Destroy COMPLETE
[Rank 0] Success
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

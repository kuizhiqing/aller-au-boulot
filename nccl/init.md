
# NCCL init

## 

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

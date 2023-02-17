# Gloo

## TL;DR;

Gloo 是集合通信和算法库。运行逻辑如下，

* 定义网络设备信息创建通信媒介 Device;
* 使用可以共同交换数据的 Store 实现 rendezvous;
* 节点通过 (rank, size) 创建 Context, 使用 connectFullMesh 方法建立通信域：
    * 向 Store 注册自身节点信息；
    * 从 Store 中获取全部节点信息；
    * 启动 n-1 个服务分别与其余节点建立连接;
* 通过 Context 使用集合通信功能。

## Concepts

**Device**

Device 是通信设备的抽象。

**Context**

Context 负责建立管理通信域。

**Store** 

Store 是实现 rendezvous 的介质，是实现共识的媒介。

**Address** 

`gloo::transport::tcp::Address` 是对网络地址的封装。

**attr**

`gloo::transport::tcp::attr` 是对网络通信属性的结构体封装，ip、网卡等信息。

## Gloo v.s. MPI

## More

ibverbs 是使用 IB(InfiniBand) 的 API.

## Workflow


```cpp
int main(void) {

  // Device
  gloo::transport::tcp::attr attr;
  auto dev = gloo::transport::tcp::CreateDevice(attr);
  // Store
  auto fileStore = gloo::rendezvous::FileStore("/tmp");
  auto prefixStore = gloo::rendezvous::PrefixStore(prefix, fileStore);
  // Context
  auto context = std::make_shared<gloo::rendezvous::Context>(rank, size);

  context->connectFullMesh(prefixStore, dev);
}
```

```cpp
// gloo/rendezvous/context.cc

void Context::connectFullMesh(
    rendezvous::Store& store,
    std::shared_ptr<transport::Device>& dev) {

  // localKey = 'rank_0'
  // value = 'host-0'
  store.set(localKey, value);
  for (int i = 0; i < size; i++) {
    // i != rank
    auto val = store.get(key);
  }

  auto transportContext = dev->createContext(rank, size);
  for (int i = 0; i < size; i++) {
    // i != rank
    auto& pair = transportContext->createPair(i);
  }

  // '0'
  // [host-0]:9000[host-1]:9001
  store.set(storeKey.str(), allBytes);

  for (int i = 0; i < size; i++) {
    // i != rank
    store.wait({key.str()}, getTimeout());
    auto allAddrs = store.get(key.str());
    auto addr = extractAddress(allAddrs, i);
    transportContext->getPair(i)->connect(addr);
  }
}

```

```cpp
std::unique_ptr<transport::Pair>& Context::createPair(int rank) {
  pairs_[rank] = std::unique_ptr<transport::Pair>(
      new tcp::Pair(this, device_.get(), rank, getTimeout()));
  return pairs_[rank];
}
```

创建 Pair 对象会在本地启动 tcp socket 服务。

```cpp
// gloo/transport/tcp/pair.cc

Pair::Pair(
    Context* context,
    Device* device,
    int rank,
    std::chrono::milliseconds timeout) {
  listen();
}

void Pair::listen() {
  std::lock_guard<std::mutex> lock(m_);
  int rv;

  const auto& attr = device_->attr_;
  auto fd = socket(attr.ai_family, attr.ai_socktype, attr.ai_protocol);
  rv = bind(fd, (const sockaddr*)&attr.ai_addr, attr.ai_addrlen);
  fd_ = fd;
  rv = ::listen(fd_, 1);
  self_ = Address::fromSockName(fd);

  device_->registerDescriptor(fd_, EPOLLIN, this);

  return;
}
```

## Reference

* [gloo](https://github.com/facebookincubator/gloo)
* [pygloo](https://github.com/ray-project/pygloo)

# Develop

这里列的 horovod 对象是用于二次开发的接口。

## Horovod 对象

```cpp
# horovod/common/common.h

enum Framework { TENSORFLOW, PYTORCH, MXNET, XLA };
enum StatusType { OK, UNKNOWN_ERROR, PRECONDITION_ERROR, ABORTED, INVALID_ARGUMENT, IN_PROGRESS };
enum DeviceType { CPU, GPU };

// for gpu
struct Event {
  std::shared_ptr<gpuEvent_t> event;
  uint64_t event_idx;
  gpuStream_t stream = nullptr;
};

class Status {
public:
  Status();
  static Status OK();
  static Status UnknownError(const std::string& message);
  static Status PreconditionError(const std::string& message);
  static Status Aborted(const std::string& message);
  static Status InvalidArgument(const std::string& message);
  static Status InProgress();
  bool ok() const;
  bool in_progress() const;
  StatusType type() const;
  const std::string& reason() const;
  Event event;

private:
  StatusType type_ = StatusType::OK;
  std::string reason_;
  Status(StatusType type, std::string reason);
};

class TensorShape {
public:
  TensorShape() : shape_() {}
  TensorShape(std::vector<int64_t> vec) : shape_(vec) {}
  void AddDim(int64_t dim);
  void AppendShape(TensorShape& other);

  std::string DebugString() const;
  int dims() const;
  int64_t dim_size(int idx) const;
  int64_t num_elements() const;
  const std::vector<int64_t>& to_vector() const;

  inline bool operator==(const TensorShape& rhs) const {
    return shape_ == rhs.shape_;
  }

  inline bool operator!=(const TensorShape& rhs) const {
    return shape_ != rhs.shape_;
  }

private:
  std::vector<int64_t> shape_;
};

class ReadyEvent {};
class ReadyEventList {};

class PersistentBuffer {
public:
  virtual const void* AccessData(std::shared_ptr<OpContext> context) const = 0;
  virtual ~PersistentBuffer() = default;
};

class Tensor {
public:
  virtual const DataType dtype() const = 0;
  virtual const TensorShape shape() const = 0;
  virtual const void* data() const = 0;
  virtual int64_t size() const = 0;
  virtual ~Tensor() = default;
};

class OpContext {
public:
  // These allocators are fully synchronous, unlike TensorFlow counterparts.
  virtual Status
  AllocatePersistent(int64_t size,
                     std::shared_ptr<PersistentBuffer>* tensor) = 0;
  virtual Status AllocateOutput(TensorShape shape,
                                std::shared_ptr<Tensor>* tensor,
                                std::shared_ptr<ReadyEvent>* event = nullptr) = 0;
  virtual Status AllocateOutput(int output_index, TensorShape shape,
                                std::shared_ptr<Tensor>* tensor,
                                std::shared_ptr<ReadyEvent>* event = nullptr) {
    if (output_index == 0) {
      return AllocateOutput(std::move(shape), tensor);
    } else {
      throw std::logic_error("output_index != 0 not supported");
    }
  }
  virtual Status AllocateZeros(int64_t num_elements, DataType dtype,
                                std::shared_ptr<Tensor>* tensor) = 0;
  virtual Framework framework() const = 0;
  virtual ~OpContext() = default;
};

// A callback to call after the communication completes. Since the
// allreduce and allgather ops are asynchronous, this callback is what resumes
// computation after the reduction is completed.
using StatusCallback = std::function<void(const Status&)>;

// Table storing Tensors to be reduced, keyed by unique name.
// This table contains everything necessary to do the distributed operation.
struct TensorTableEntry {
  std::string tensor_name;
  std::shared_ptr<OpContext> context;
  std::shared_ptr<Tensor> tensor;
  std::shared_ptr<Tensor> output;
  // Identifier for the subset of Horovod processes partaking in this operation.
  int32_t process_set_id = 0;
  // Root rank for broadcast operation (relative to process set).
  int root_rank = 0;
  // List of events indicating that data is ready.
  ReadyEventList ready_event_list;
  // GPU to do reduction on, or CPU_DEVICE_ID in case of CPU.
  int device = CPU_DEVICE_ID;
  // A callback to call with the status.
  StatusCallback callback;
  // If we build with NVTX support: A range marking the start
  // and end of the distributed op for this tensor (may be
  // shared by multiple tensors).
  SharedNvtxOpRange nvtx_op_range;

  // Alltoall splits (if tensor is for an Alltoall operation)
  // Note: splits are stored in TensorTableEntry to avoid N^2
  // storage complexity of collecting all worker split arrays
  // on coordinator rank.
  std::vector<int32_t> splits;
  std::shared_ptr<Tensor> received_splits;

  void FinishWithCallback(const Status& status);
};
```

### Message 

#### Request 

```cpp
// horovod/common/message.h

// Request 是非 0 worker 向 0 号 worker 即 coordinator 发送消息的消息体
class Request {
  enum RequestType {
    ALLREDUCE = 0,
    ALLGATHER = 1,
    BROADCAST = 2,
    JOIN = 3,
    ADASUM = 4,
    ALLTOALL = 5,
    BARRIER = 6,
    REDUCESCATTER = 7
  };

  static const std::string& RequestType_Name(RequestType value);
  int32_t request_rank() const;
  void set_request_rank(int32_t value);
  RequestType request_type() const;
  void set_request_type(RequestType value);
  DataType tensor_type() const;
  void set_tensor_type(DataType value);
  const std::string& tensor_name() const;
  void set_tensor_name(const std::string& value);
  int32_t root_rank() const;
  void set_root_rank(int32_t value);
  int32_t device() const;
  void set_device(int32_t value);
  int32_t group_id() const;
  void set_group_id(int32_t value);
  const std::vector<int64_t>& tensor_shape() const;
  void set_tensor_shape(const std::vector<int64_t>& value);
  void add_tensor_shape(int64_t value);
  double prescale_factor() const;
  double postscale_factor() const;
  void set_prescale_factor(double prescale_factor);
  void set_postscale_factor(double postscale_factor);
  static void ParseFromBytes(Request& request, const uint8_t* input);
  static void SerializeToString(const Request& request, std::string& output);

  int32_t request_rank_ = 0;
  RequestType request_type_ = RequestType::ALLREDUCE;
  DataType tensor_type_ = DataType::HOROVOD_UINT8;
  int32_t root_rank_ = 0;
  int32_t device_ = 0;
  int32_t group_id_ = NULL_GROUP_ID;
  std::string tensor_name_;
  std::vector<int64_t> tensor_shape_;
  double prescale_factor_ = 1.0;
  double postscale_factor_ = 1.0;
};

class RequestList {
  const std::vector<Request>& requests() const;
  void set_requests(const std::vector<Request>& value);
  void add_request(const Request& value);
  void emplace_request(Request&& value);
  bool shutdown() const;
  void set_shutdown(bool value);
  static void ParseFromBytes(RequestList& request_list, const uint8_t* input);
  static void SerializeToString(const RequestList& request_list, std::string& output);

  std::vector<Request> requests_;
  bool shutdown_ = false;
};
```

#### Response  

```cpp
// Response 是 0 号 worker 即 coordinator 向非 0 worker 发送消息的消息体
class Response {
public:
  enum ResponseType {
    ALLREDUCE = 0,
    ALLGATHER = 1,
    BROADCAST = 2,
    JOIN = 3,
    ADASUM = 4,
    ALLTOALL = 5,
    BARRIER = 6,
    REDUCESCATTER = 7,
    ERROR = 8
  };

  static const std::string& ResponseType_Name(ResponseType value);
  ResponseType response_type() const;
  void set_response_type(ResponseType value);
  const std::vector<std::string>& tensor_names() const;
  DataType tensor_type() const;
  void set_tensor_type(DataType value);
  const std::string tensor_names_string() const;
  void set_tensor_names(const std::vector<std::string>& value);
  void add_tensor_name(const std::string& value);
  void add_tensor_name(std::string&& value);
  const std::string& error_message() const;
  void set_error_message(const std::string& value);
  const std::vector<int32_t>& devices() const;
  void set_devices(const std::vector<int32_t>& value);
  void add_device(int32_t value);
  const std::vector<int64_t>& tensor_sizes() const;
  void set_tensor_sizes(const std::vector<int64_t>& value);
  void add_tensor_size(int64_t value);
  void add_allgather_response(const Response& response);
  double prescale_factor() const;
  double postscale_factor() const;
  void set_prescale_factor(double prescale_factor);
  void set_postscale_factor(double postscale_factor);
  int last_joined_rank() const;
  void set_last_joined_rank(int value);
  static void ParseFromBytes(Response& response, const uint8_t* input);
  static void SerializeToString(const Response& response, std::string& output);

  ResponseType response_type_ = ResponseType::ALLREDUCE;
  std::vector<std::string> tensor_names_;
  DataType tensor_type_ = DataType::HOROVOD_UINT8;
  std::string error_message_;
  std::vector<int32_t> devices_;
  std::vector<int64_t> tensor_sizes_;
  double prescale_factor_ = 1.0;
  double postscale_factor_ = 1.0;
  int last_joined_rank_ = -1;
};

class ResponseList {
  const std::vector<Response>& responses() const;
  void set_responses(const std::vector<Response>& value);
  void add_response(const Response& value);
  void add_response(Response&& value);
  void emplace_response(Response&& value);
  bool shutdown() const;
  void set_shutdown(bool value);
  static void ParseFromBytes(ResponseList& response_list, const uint8_t* input);
  static void SerializeToString(const ResponseList& response_list, std::string& output);

  std::vector<Response> responses_;
  bool shutdown_ = false;
};
```
ResponseCache 

```cpp
// horovod/common/response_cache.h

struct TensorParams {
  DataType dtype;
  std::vector<int64_t> shape;
  int32_t device;
};

// LRU cache of Responses
class ResponseCache {
public:
  ResponseCache() = default;
  ResponseCache(const ResponseCache&) = delete;

  enum CacheState { MISS = 0, HIT = 1, INVALID = 2 };

  void clear();

  void set_capacity(uint32_t capacity);

  uint32_t capacity() const;

  size_t num_active_bits() const;

  CacheState cached(const Request& message) const;

  CacheState cached(const Response& response, const TensorParams& params,
                    bool joined = false) const;

  void put(const Response& response, TensorQueue& tensor_queue,
           bool joined = false);

  const Response& get_response(uint32_t cache_bit);

  const Response& peek_response(uint32_t cache_bit) const;

  uint32_t peek_cache_bit(const Request& message) const;

  uint32_t peek_cache_bit(const std::string& tensor_name) const;

  std::vector<uint32_t> list_all_bits() const;

  void erase_response(uint32_t cache_bit);

  void update_cache_bits();

private:
  void put_(const Response& response, TensorParams& params,
            bool joined = false);

  uint32_t capacity_ = 0;

  // List containing cached entries. Each entry in the cache is a pair
  // of a Response and a TensorParams struct.
  std::list<std::pair<Response, TensorParams>> cache_;

  // Vector of iterators to cache entries. Indexed by cache bit.
  std::vector<std::list<std::pair<Response, TensorParams>>::iterator>
      cache_iters_;

  // Lookup table mapping tensor names to assigned cache bits.
  std::unordered_map<std::string, uint32_t> tensor_name_to_bit_;

  bool bits_outdated_ = false;

  bool print_warning_ = true;
};

// Helper class to coordinate cache and state information
// across workers. Uses global controller operations on a bit vector
// for cheaper coordination.
class CacheCoordinator {
public:
  explicit CacheCoordinator(size_t num_active_bits_);

  void record_hit(uint32_t bit);

  void record_invalid_bit(uint32_t bit);

  void erase_hit(uint32_t bit);

  void set_should_shut_down(bool should_shut_down);

  void set_uncached_in_queue(bool uncached_in_queue);

  const std::set<uint32_t>& cache_hits() const;

  const std::set<uint32_t>& invalid_bits() const;

  const std::set<uint32_t>& timeline_bits() const;

  bool should_shut_down() const;

  bool uncached_in_queue() const;

  // Method to sync state and bit sets across workers.
  void sync(std::shared_ptr<Controller> controller, bool timeline_enabled);

private:
  enum StatusBit {
    SHOULD_SHUT_DOWN = 0,
    UNCACHED_IN_QUEUE = 1,
    INVALID_IN_QUEUE = 2
  };

  // Number of active bits in the cache. Required to size the
  // bitvector identically across workers.
  size_t num_active_bits_;

  // Set of cache hit bits. After sync(), contains only common
  // cache hit bits across workers.
  std::set<uint32_t> cache_hits_;

  // Set of invalid bits. After sync(), contains only common
  // invalid bits across workers.
  std::set<uint32_t> invalid_bits_;

  // Set of bits for timeline handling. After sync(), contains bits
  // where at least one worker recorded a cache hit. This indicates
  // that the timeline negotion phase should be started/continued.
  std::set<uint32_t> timeline_bits_;

  // States used externally in cycle loop.
  bool should_shut_down_ = false;
  bool uncached_in_queue_ = false;

  // State used internally to trigger second bit vector communication
  // to sync invalid bits.
  bool invalid_in_queue_ = false;

  std::vector<long long> bitvector_;

  bool synced_ = false;
};
```

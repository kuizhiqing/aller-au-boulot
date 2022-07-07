# common

Horovod 对象
```cpp
# horovod/common/common.h

enum Framework { TENSORFLOW, PYTORCH, MXNET, XLA };
enum StatusType { OK, UNKNOWN_ERROR, PRECONDITION_ERROR, ABORTED, INVALID_ARGUMENT, IN_PROGRESS };
enum DeviceType { CPU, GPU };

// for gpu
struct Event {};

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

# Tensor

## Tensor API dependency

The dependency of tensor related API

![Tensor View API Dependence](tensor-api.png)


## View

Tensor view explication

![Tensor View API Dependence](tensor-view.png)


## Data Structure

```cpp
// torch/csrc/Module.cpp

extern "C"
TORCH_API PyObject* initModule();
PyObject* initModule() {
  THPUtils_addPyMethodDefs(methods, TorchMethods);

  static struct PyModuleDef torchmodule = {
     PyModuleDef_HEAD_INIT,
     "torch._C",
     nullptr,
     -1,
     methods.data()
  };
  ASSERT_TRUE(module = PyModule_Create(&torchmodule));
  ASSERT_TRUE(THPGenerator_init(module));
  ASSERT_TRUE(THPException_init(module));
  THPSize_init(module);
  THPDtype_init(module);
  THPDTypeInfo_init(module);
  THPLayout_init(module);
  THPMemoryFormat_init(module);
  THPQScheme_init(module);
  THPDevice_init(module);
  THPStream_init(module);
  ASSERT_TRUE(THPVariable_initModule(module));
  ASSERT_TRUE(THPFunction_initModule(module));
  ASSERT_TRUE(THPEngine_initModule(module));

  at::init();

  return module;
}
```

```cpp
// torch/csrc/autograd/python_variable.cpp

bool THPVariable_initModule(PyObject *module)
{
  THPVariableMetaType.tp_base = &PyType_Type;
  if (PyType_Ready(&THPVariableMetaType) < 0)
    return false;
  Py_INCREF(&THPVariableMetaType);
  PyModule_AddObject(module, "_TensorMeta",   (PyObject *)&THPVariableMetaType);

  static std::vector<PyMethodDef> methods;
  THPUtils_addPyMethodDefs(methods, torch::autograd::variable_methods);
  THPUtils_addPyMethodDefs(methods, extra_methods);
  THPVariableType.tp_methods = methods.data();
  if (PyType_Ready(&THPVariableType) < 0)
    return false;
  Py_INCREF(&THPVariableType);
  PyModule_AddObject(module, "_TensorBase",   (PyObject *)&THPVariableType);
  torch::autograd::initTorchFunctions(module);
  torch::autograd::initTensorImplConversion(module);
  return true;
}

PyTypeObject THPVariableType = {
    PyVarObject_HEAD_INIT(
        &THPVariableMetaType,
        0) "torch._C._TensorBase", /* tp_name */
    sizeof(THPVariable), /* tp_basicsize */
    0, /* tp_itemsize */
    ...
    THPVariable_pynew, /* tp_new */
};

PyTypeObject THPVariableMetaType = {
  PyVarObject_HEAD_INIT(DEFERRED_ADDRESS(&PyType_Type), 0)
  "torch._C._TensorMeta",                      /* tp_name */
  sizeof(THPVariableMeta),
  ...
  THPVariableMetaType_init,                    /* tp_init */
  nullptr,                                     /* tp_alloc */
  nullptr,                                     /* tp_new */
};
```

python 的 `_TensorBase` 绑定在 `THPVariableType` 上，python 相关的 `Tensor` 都继承自 `torch._C._TensorBase`.

```cpp
// torch/csrc/autograd/python_variable.h

// Python object that backs torch.autograd.Variable
// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
struct THPVariable {
  PyObject_HEAD;
  // Payload
  c10::MaybeOwned<at::Tensor> cdata;
  // Hooks to be run on backwards pass (corresponds to Python attr
  // '_backwards_hooks', set by 'register_hook')
  PyObject* backward_hooks = nullptr;
};
```

```cpp
// torch/csrc/autograd/function_hook.h

namespace torch { namespace autograd {

using Variable = at::Tensor;
using variable_list = std::vector<Variable>;

} }
```

PyTorch Tensor 相关主要数据结构和关系

![1](1.png)

Tensor v.s. TensorBase

```cpp
// aten/src/ATen/core/TensorBase.h

// Convert Tensor to TensorBase without any need to include Tensor.h
TORCH_API const TensorBase& get_tensor_base(const Tensor& t);

// NOTE: [Tensor vs. TensorBase]
//
// Tensor, being the central data structure in PyTorch, gets used and
// it's header included almost everywhere. Unfortunately this means
// every time an operator signature is updated or changed in
// native_functions.yaml, you (and every other PyTorch developer) need
// to recompile all of ATen and it's dependencies.
//
// TensorBase aims to break up these header dependencies, and improve
// incremental build times for all PyTorch developers. TensorBase
// represents a reference counted handle to TensorImpl, exactly the
// same as Tensor. However, TensorBase doesn't have code generated
// methods in it's API and thus no dependence on native_functions.yaml.
//
// Usage tips
// ----------
// - You can `#define TORCH_ASSERT_NO_OPERATORS` at the top of a .cpp
//   or .cu file to ensure it has no header dependencies on
//   native_functions.yaml (direct or indirect).
// - Tensor inherits from TensorBase, so functions taking
//   `const TensorBase &` are callable with Tensor as well.
// - TensorBase can be converted to tensor with `Tensor(tensor_base)`,
//   but this requires a reference-count bump. OptionalTensorRef on
//   the other hand can materialize a `const Tensor &` without
//   touching the reference-count.
class TORCH_API TensorBase {
 public:
  const c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>& getIntrusivePtr() const {
    return impl_;
  }

  c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl> unsafeReleaseIntrusivePtr() {
    return std::move(impl_);
  }

  DispatchKeySet key_set() const {
    return impl_->key_set();
  }
  ScalarType scalar_type() const {
    return typeMetaToScalarType(impl_->dtype());
  }
  const Storage& storage() const {
    return impl_->storage();
  }

  Layout layout() const noexcept {
    return impl_->layout();
  }

  caffe2::TypeMeta dtype() const noexcept {
    return impl_->dtype();
  }

  inline Device device() const {
    return impl_->device();
  }

  int64_t get_device() const {
    // NB: this is not a native function to avoid dispatching overhead.
    return impl_->get_device();
  }

  TensorOptions options() const {
    return TensorOptions().dtype(dtype())
                          .device(device())
                          .layout(layout());
  }

  void* data_ptr() const {
    return this->unsafeGetTensorImpl()->data();
  }

  template <typename T>
  T * data_ptr() const;

  at::TensorBase tensor_data() const;

  at::TensorBase variable_data() const;

  const std::shared_ptr<torch::autograd::Node>& grad_fn() const;

protected:
  c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl> impl_;
};
```

`Tensor` 继承自 `TensorBase`, `TensorBase` 不依赖 function 自动生成，使用 `TensorBase` 可以避免自动生成部分有改动时全量编译，其次是引用计数问题。

Tensor 的继承关系

![1](2.png)

`at:Tensor` 本质是 Tensor 的 API，底层是 `TensorImpl`

```cpp
// c10/core/TensorImpl.h

/**
 * The low-level representation of a tensor, which contains a pointer
 * to a storage (which contains the actual data) and metadata (e.g., sizes and
 * strides) describing this particular view of the data as a tensor.
 *
 */

struct C10_API TensorImpl : public c10::intrusive_ptr_target {
  enum ImplType { VIEW };

 public:

  virtual IntArrayRef strides() const;

  TENSORIMPL_MAYBE_VIRTUAL const Storage& storage() const {
    return storage_;
  }

  Device device() const {
    return *device_opt_;
  }

  Layout layout() const {
    // This keyset must also be kept in sync with the logic in
    // is_sparse() / is_sparse_csr() / is_mkldnn()
    constexpr auto sparse_and_sparsecsr_and_mkldnn_ks =
        c10::sparse_ks | c10::sparse_csr_ks | c10::mkldnn_ks;
    ...
  }

  Storage storage_;

  inline T* data() const {
      return data_ptr_impl<T>();
  }
  inline T* data_ptr_impl() const {
      return storage_.unsafe_data<T>() + storage_offset_;
  }

  inline void* data() const {
      return static_cast<void*>(
        static_cast<char*>(storage_.data()) +
        data_type_.itemsize() * storage_offset_);
  }

  const caffe2::TypeMeta dtype() const {
    return data_type_;
  }

  DeviceType device_type() const {
    return (*device_opt_).type();
  }

 private:
  // This pointer points to an AutogradMeta struct that stores autograd-specific
  // fields (such as grad_ / grad_fn_ / grad_accumulator_). This pointer always
  // has unique ownership (meaning only one TensorImpl can own it at a time).
  //
  std::unique_ptr<c10::AutogradMetaInterface> autograd_meta_ = nullptr;

 protected:
  std::unique_ptr<c10::NamedTensorMetaInterface> named_tensor_meta_ = nullptr;

  c10::VariableVersion version_counter_;

  PyObject* pyobj_;

  c10::impl::SizesAndStrides sizes_and_strides_;

  caffe2::TypeMeta data_type_;

  c10::optional<c10::Device> device_opt_;

  const at::Tensor& grad() const;
}
```

Storage

```cpp
// c10/core/Storage.h

struct C10_API Storage {

  void* data() const {
    return storage_impl_.get()->data();
  }

  at::DataPtr& data_ptr() {
    return storage_impl_->data_ptr();
  }

  const at::DataPtr& data_ptr() const {
    return storage_impl_->data_ptr();
  }

 protected:
  c10::intrusive_ptr<StorageImpl> storage_impl_;
};
```

StorageImpl

```cpp
// c10/core/StorageImpl.h 

struct C10_API StorageImpl : public c10::intrusive_ptr_target {
 private:
  DataPtr data_ptr_;
  size_t size_bytes_;
  bool resizable_;
  bool received_cuda_;
  Allocator* allocator_;
```

UniqueVoidPtr 

```cpp
// c10/util/UniqueVoidPtr.h

namespace c10 {

using DeleterFnPtr = void (*)(void*);

namespace detail {

class UniqueVoidPtr {
 private:
  void* data_;
  std::unique_ptr<void, DeleterFnPtr> ctx_;
}

} }
```

`detail::UniqueVoidPtr` is an owning smart pointer like `unique_ptr`

* specialized to void 
* specialized for a function pointer deleter `void(void* ctx)`
* deleter is guaranteed to be called when the unique pointer is destructed and the context is non-null


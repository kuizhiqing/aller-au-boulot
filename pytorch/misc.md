# PyTorch

## 目录结构 code base structure

```yaml
c10: core library; essential functionality; (moving ATen/core to it)
aten: cpp tensor library (no autograd)
  src:
    ATen:
      - core: moving above
      - native: operators
torch: python library (including csrc)
  csrc:
    - autograd: automatic differenctiation
    - api: cpp api
    - distributed: distributed training
tools: code generation script
```

## View

`narrow()`, `view()`, `expand()` and `transpose()`

```yaml
# aten/src/ATen/native/TensorShape.cpp



split: narrow

  narrow: slice

    slice: as_strided


```

```
.stride()
.shape .size()
.contiguous() 
.is_contiguous()
```

### narrow

```cpp
// aten/src/ATen/BatchingRegistrations.cpp

TORCH_LIBRARY_IMPL(aten, Batched, m) {
  // NB: static_cast because there's another variant of narrow. However, we don't
  // want to support the other variant yet bc it isn't documented...
  m.impl("narrow", static_cast<Tensor(*)(const Tensor&,int64_t,int64_t,int64_t)>(native::narrow)); // composite wrt autograd
}
```

```cpp
// aten/src/ATen/native/TensorShape.cpp

Tensor narrow(const Tensor& self, int64_t dim, int64_t start, int64_t length) {
  TORCH_CHECK(self.dim() > 0, "narrow() cannot be applied to a 0-dim tensor.");
  auto cur_size = self.size(dim);
  if (start != cur_size) {  // start being the end is valid, but not a valid dim specification.
    start = maybe_wrap_dim(start, cur_size);
  }
  TORCH_CHECK(length >= 0 && start <= cur_size - length,
           "start (", start, ") + length (", length, ") exceeds dimension size (", cur_size, ").");
  return at::slice(self, dim, start, start + length, 1);
}

Tensor narrow(const Tensor& self, int64_t dim, const Tensor& start, int64_t length) {
  TORCH_CHECK(start.dim() == 0 && isIntegralType(start.scalar_type(), /*includeBool=*/false),
              "start must be an 0-dim integral Tensor.");
  int64_t st = start.item<int64_t>();
  return at::narrow(self, dim, st, length);
}
```

内部使用 `slice` 函数，这个函数并没有 python 绑定。

```cpp
// build/aten/src/ATen/RegisterFunctionalization_0.cpp

    at::Tensor as_strided(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef size, at::IntArrayRef stride, c10::optional<int64_t>   storage_offset) {

      auto self_ = at::functionalization::impl::from_functional_tensor(self);
      at::Tensor tmp_output;
      at::Tensor reference_tensor_output;
      {    
        at::AutoDispatchSkipFunctionalize guard;
        auto self_meta = at::native::empty_strided_meta(self.sizes(), self.strides(), /*dtype=*/c10::make_optional(self.scalar_type()), /*layout=*/c10::      make_optional(self.layout()), /*device=*/c10::make_optional(c10::Device(kMeta)), /*pin_memory=*/c10::nullopt);
        reference_tensor_output = at::_ops::as_strided::call(self_meta, size, stride, storage_offset);
        tmp_output = at::_ops::as_strided::redispatch(dispatchKeySet & c10::after_func_keyset, self_, size, stride, storage_offset);
        // I'm fusing the [alias removal], [mutation removal], [add views back] passes together.
        // Later, we'll want to turn them into separate passes (since e.g. vulkan only cares about alias removal).
      }    
      at::functionalization::ViewMeta view_meta = at::functionalization::ViewMeta(
        [size = size.vec(), stride = stride.vec(), storage_offset = storage_offset](const at::Tensor & base, int64_t mutated_view_idx) -> at::Tensor {
          return at::_ops::as_strided::call(base, size, stride, storage_offset);
        },   
        [size = size.vec(), stride = stride.vec(), storage_offset = storage_offset](const at::Tensor & base, const at::Tensor & mutated_view, int64_t         mutated_view_idx) -> at::Tensor {
          return at::functionalization::FunctionalInverses::as_strided_inverse(base, mutated_view, size, stride, storage_offset);
        }    
      );   
      auto out = at::functionalization::impl::create_functional_tensor_with_view_meta(tmp_output, self, view_meta);
      // See  Note [Propagating strides in the functionalization pass]
      at::functionalization::impl::set_sizes_strides_offset(out, reference_tensor_output);
      return out; 

    }
```

## Multiprocessing

定义在 `torch/_tensor.py` 中的 `subclassed class` 如果需要跨进程，需要修改`torch/multiprocessing/reductions.py` 中的 `ForkingPickler` 做序列化。

## 

## RPC 分布式

#### rpc 函数

* rpc_async() 

* rpc_sync()

* remote(), no return e.x. create table from driver

#### Remote reference ：分布式共享指针

#### Distributed Autograd

#### Distibuted Optimizer

# Hook


### Summary

| Hook | 触发时机 | API |
| ---------- | ---------- | --------- |
| 反向 hook | 反向梯度计算完成后 | register_hook(hook) | 
| 前向 after hook| forward() 调用之后 | register_forward_hook(hook) | 
| 前向 before hook | forward() 调用之前 | register_forward_pre_hook(hook) | 
| module hook | module inputs 反向梯度计算完成后 | register_full_backward_hook(hook) | 
| 节点 hook | autograd hook, 节点计算后 | register_hook(hook) | 
| 通信 hook | hook 参数 dist.GradBucket ready | register_comm_hook(state, hook) | 

> torch.nn.parallel.DistributedDataParallel.register_comm_hook
Tensor.register_hook(hook)
torch.nn.Module.register_forward_hook(hook)
torch.nn.Module.register_forward_pre_hook(hook)
torch.nn.Module.register_full_backward_hook(hook)
Node.register_hook(hook)
torch.nn.parallel.DistributedDataParallel.register_comm_hook(state, hook)


### 反向 hook

反向 hook 最为常用的 hook，但反向梯度生成后加入逻辑。

```python
# hook signature
hook(grad) -> Tensor or None

def hook(grad):
    return grad * 2

v = torch.tensor([0., 0., 0.], requires_grad=True)
h = v.register_hook(hook)  # double the gradient
v.backward(torch.tensor([1., 2., 3.]))
v.grad # 2 4 6
h.remove()  # removes the hook
```

### Module hook
```python
register_forward_hook(hook)
hook(module, input, output) -> None or modified output
register_forward_pre_hook(hook)
hook(module, input) -> None or modified input
register_full_backward_hook(hook)
hook(module, grad_input, grad_output) -> tuple(Tensor) or None
# 不允许修改参数 tensor，但是可以返回 tensor 用于后续梯度计算
```

### Autograd Hook
autograd hook，node grad
经典用法来自 horovod optimizer

```python
p_tmp = p.expand_as(p)
grad_acc = p_tmp.grad_fn.next_functions[0][0]
grad_acc.register_hook(self._make_hook(p))
# 示例
>>> p = torch.tensor([1,2,3], dtype=float, requires_grad=True)
>>> p
tensor([1., 2., 3.], dtype=torch.float64, requires_grad=True)
>>> pt = p.expand_as(p)
>>> pt
tensor([1., 2., 3.], dtype=torch.float64, grad_fn=<ExpandBackward0>)
>>> pg = pt.grad_fn.next_functions[0][0]
>>> pg
<AccumulateGrad object at 0x7fa7ade03710>
```

### DDP 通信 hook

```python
torch.nn.parallel.DistributedDataParallel.register_comm_hook(state, hook)
hook(state: object, bucket: dist.GradBucket) -> torch.futures.Future[torch.Tensor]
```

**Example**
```python
def encode_and_decode(state: object, bucket: dist.GradBucket): -> torch.futures.Future[torch.Tensor]
    encoded_tensor = encode(bucket.buffer()) # encode gradients
    fut = torch.distributed.all_reduce(encoded_tensor).get_future()
    # Define the then callback to decode.
    def decode(fut):
        decoded_tensor = decode(fut.value()[0]) # decode gradients
        return decoded_tensor
    return fut.then(decode)
```


### Reference

* https://pytorch.org/docs/stable/generated/torch.Tensor.register_hook.html
* https://pytorch.org/docs/stable/ddp_comm_hooks.html





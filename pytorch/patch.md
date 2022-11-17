# PyTorch Patch

## summary

这个库提供了一种对 pytorch 进行 patch/hack 的方式，在很多场景可以借鉴，例如容错弹性。

```python
patch_apex()
patch_torch_classes() # [torch, torch.Tensor, torch.nn.functional, torch.distributed]
patch_torch_nn_forward_functions() # [torch.nn.RNN, torch.nn.RNNCell, torch.nn.LSTM, torch.nn.LSTMCell, torch.nn.GRU, torch.nn.GRUCell]
```

```python
patch_apex --> patch_apex_c + patch_apex_pyt
    patch_apex_c --> patchClass --> add_wrapper
    patch_apex_pyt --> patch_apex_module --> patch_apex_class --> add_wrapper

patch_torch_classes --> patchClass --> add_wrapper
patch_torch_nn_forward_functions --> add_wrapper
```

### patch_apex

```python
def patch_apex():
    patch_apex_c()
    patch_apex_pyt()

def patch_apex_c():
    if importlib.util.find_spec("amp_C") is not None:
        import amp_C
        patchClass(amp_C)
    # fused_adam_cuda
    # fused_lamb_cuda
    # fused_layer_norm_cuda
    # distributed_lamb_cuda
    # xentropy_cuda
    # mlp_cuda

def patch_apex_pyt():
    if importlib.util.find_spec("apex") is not None:
        patch_apex_module("apex.amp")
        patch_apex_module("apex.contrib.groupbn")
        patch_apex_module("apex.contrib.multihead_attn")
        patch_apex_module("apex.contrib.optimizers")
        patch_apex_module("apex.contrib.sparsity")
        patch_apex_module("apex.contrib.xentropy")
        patch_apex_module("apex.fp16_utils")
        patch_apex_module("apex.mlp")
        patch_apex_module("apex.multi_tensor_apply")
        patch_apex_module("apex.optimizers")
        patch_apex_module("apex.parallel")

def patch_apex_module(modstr):
    """ 
    Patch all forward/backward/step functions in classes in the given apex module.
    """
    if importlib.util.find_spec(modstr) is not None:
        mod = importlib.import_module(modstr)

        for _, v in ins.getmembers(mod):
            # This makes sure we don't patch random other modules that are imported by the target module
            if is_same_module_or_submodule(mod, ins.getmodule(v)):
                if (ins.isclass(v)):
                    patch_apex_class(v)

def patch_apex_class(cls):
    """
    Patch all forward/backward/step functions in the given apex class
    """
    for f in cls.__dict__:
        if (ins.isfunction(cls.__dict__[f])):
            if f in ["forward", "backward", "step"]:
                add_wrapper(cls, f)
```

### patch_torch_classes

```python
def patchClass(cls):
    for f in dir(cls):
        if isfunc(cls, f):
            add_wrapper(cls, f)


def patch_torch_classes():
    """Monkey-patch all classes in torch"""
    for cls in [torch, torch.Tensor, torch.nn.functional, torch.distributed]:
        patchClass(cls)
```

### patch_torch_nn_forward_functions

```python
def patch_torch_nn_forward_functions():
    """Monkey-patch all forward functions in torch.nn libraries"""
    for cls in [torch.nn.RNN, torch.nn.RNNCell, torch.nn.LSTM, torch.nn.LSTMCell, torch.nn.GRU, torch.nn.GRUCell]:
        if isfunc(cls, 'forward'):
            add_wrapper(cls, 'forward')
```

### add_wrapper

```python
def add_wrapper(mod, fn_name):

    # Get a pointer to the original function
    func = getattr(mod, fn_name)

    # Check if the mod has a string representation
    # and is not a Script or Traced module (used by JIT)
    # yapf: disable
    s = hasattr(mod, "extra_repr") and (type(mod) is not torch.jit.ScriptModule
                                       ) and (type(mod) is not torch.jit.TopLevelTracedModule)
    # yapf: enable

    def wrapper_func(*args, **kwargs):

        # Extract the stacktrace
        stack = traceback.extract_stack()

        # Push trace marker
        nvtx.range_push(traceMarker(stack))

        # Push module marker
        if s:
            m = modMarker(mod, fn_name, args)
            nvtx.range_push(m)

        # Create and push argument marker
        cadena = argMarker(mod, fn_name, args, kwargs)
        nvtx.range_push(cadena)

        # Call the original function
        result = func(*args, **kwargs)

        # Pop argumet marker
        nvtx.range_pop()

        # Pop module marker
        if s:
            nvtx.range_pop()

        # Pop trace marker
        nvtx.range_pop()

        return result

    setattr(mod, fn_name, wrapper_func)
```



## Reference

* https://github.com/NVIDIA/PyProf.git

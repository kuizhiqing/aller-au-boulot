# TVM

## Expression

**TE**
Tensor Expression, DSL (domain-specific language)

**Relay** TVM’s high-level model language

**IR** intermediate representation

**TIR** Tensor Intermediate Representation, TVM’s low-level intermediate representation

**Schedule** how to execute the computation

**Stage** schedule for one operation

**TOPI** TVM Operator Inventory, numpy-style generic operations and schedules 

## Demo

```python
from tvm.driver import tvmc

#Step 1: Load
model = tvmc.load('resnet50-v2-7.onnx', shape_dict={'data':[1, 3, 224, 224]}) 

#Step 1.5: Optional Tune
tuning_records = tvmc.tune(model, target="llvm") 

#Step 2: Compile
package = tvmc.compile(model, target="llvm", tuning_records = tuning_records) 

#Step 3: Run
result = tvmc.run(package, device="cpu") 

print(result)
```

* IRModule: relay.Function + tir.PrimFunc
* tvmc.compile: relay::Function --> tir::PrimFunc

**Why TVM ?**

`tvmc.tune` make model run more fast.

**How ?**

ALL, especially,

AutoTVM (template-based) or AutoScheduler (Ansor, template-free auto-tuning)


## TVM optimizaing compiler workflow

1. TF/PyTroch/ONNX
2. Relay (High-level IR)
3. TE (Computation definition)
4. AutoTVM/AutoScheduler (Auto-tuining module)
5. TE + Schedule (Optimization specification)
6. TIR (Low-level IR)
7. Machine Code

流程解析

1. TVM 数据输入格式，prefer ONNX
2. TVM 高级 API 操作计算图
3. Relay 通过 fuseops pass 生成子图，同时有 schedule primitives 对 low-level loop 进行优化，Tensor Operator Inventory (TOPI) 处理常规 op, 生成 TE
4. 通过 AutoTVM (template-based) 或 AutoScheduler (template-free auto-tuning) 寻找最佳 schedule
5. 生成 json 格式 tuning records, 包含最佳 schedule
6. 生成 TIR，支持主流 LLVM/NVCC
7. 生成机器码


low-level loop optimizations: tiling, vectorization, parallelization, unrolling, and fusion

## TVM Auto-scheduler (a.k.a. Ansor)

package `tvm.auto_scheduler`

## Schedule Primitives

How to get good performance kernel ?

## TE

```python
import tvm
from tvm import te

m = te.var('m')
n = te.var('n')

a = te.placeholder((m, n), name='A')
b = te.placeholder((m, n), name='B')

c = te.compute((m, n), lambda i, j: a[i, j]*b[i, j], name='C')

s = te.create_schedule([c.op])

tgt = tvm.target.Target(target="llvm", host="llvm")
mult = tvm.build(s, [a, b, c], target=tgt, name="mult")

print(mult.get_source())

print(tvm.lower(s, [a, b, c], simple_mode=True))
```

`tvm.build` 

tvm.te.schedule.Schedule, tvm.tir.PrimFunc, IRModule, Mapping[str, IRModule] --> `tvm.runtime.Module`

A module that combines both host and device code

`tvm.lower` 

tvm.te.schedule.Schedule, tvm.tir.PrimFunc, IRModule --> `IRModule`

Demo for IRModule transform

```python
import tvm
from tvm.ir.module import IRModule
from tvm.script import tir as T
import numpy as np

from tvm import te

A = te.placeholder((8,), dtype="float32", name="A")
B = te.compute((8,), lambda *i: A(*i) + 1.0, name="B")
func = te.create_prim_func([A, B])
ir_module = IRModule({"main": func})
print(ir_module.script())
"""
# from tvm.script import tir as T
@tvm.script.ir_module
class Module:
    @T.prim_func
    def main(A: T.Buffer[8, "float32"], B: T.Buffer[8, "float32"]) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        # with T.block("root")
        for i0 in T.serial(8):
            with T.block("B"):
                i0_1 = T.axis.spatial(8, i0)
                T.reads(A[i0_1])
                T.writes(B[i0_1])
                B[i0_1] = A[i0_1] + T.float32(1)
"""
```

```python
# <class 'tvm.driver.build_module.OperatorModule'>
mod = tvm.build(ir_module, target="llvm")  # The module for CPU backends.

# <class 'tvm.tir.schedule.schedule.Schedule'>
sch = tvm.tir.Schedule(ir_module)
block_b = sch.get_block("B")
(i,) = sch.get_loops(block_b)
```

```python
i_0, i_1, i_2 = sch.split(i, factors=[2, 2, 2])
print(sch.mod.script())
"""
# from tvm.script import tir as T
@tvm.script.ir_module
class Module:
    @T.prim_func
    def main(A: T.Buffer[8, "float32"], B: T.Buffer[8, "float32"]) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        # with T.block("root")
        for i0_0, i0_1, i0_2 in T.grid(2, 2, 2):
            with T.block("B"):
                i0 = T.axis.spatial(8, i0_0 * 4 + i0_1 * 2 + i0_2)
                T.reads(A[i0])
                T.writes(B[i0])
                B[i0] = A[i0] + T.float32(1)
"""
```

```python
sch.reorder(i_0, i_2, i_1)
print(sch.mod.script())
"""
# from tvm.script import tir as T
@tvm.script.ir_module
class Module:
    @T.prim_func
    def main(A: T.Buffer[8, "float32"], B: T.Buffer[8, "float32"]) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        # with T.block("root")
        for i0_0, i0_2, i0_1 in T.grid(2, 2, 2):
            with T.block("B"):
                i0 = T.axis.spatial(8, i0_0 * 4 + i0_1 * 2 + i0_2)
                T.reads(A[i0])
                T.writes(B[i0])
                B[i0] = A[i0] + T.float32(1)
"""
```

```python
sch.bind(i_0, "blockIdx.x")
sch.bind(i_2, "threadIdx.x")
print(sch.mod.script())
"""
# from tvm.script import tir as T
@tvm.script.ir_module
class Module:
    @T.prim_func
    def main(A: T.Buffer[8, "float32"], B: T.Buffer[8, "float32"]) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        # with T.block("root")
        for i0_0 in T.thread_binding(2, thread="blockIdx.x"):
            for i0_2 in T.thread_binding(2, thread="threadIdx.x"):
                for i0_1 in T.serial(2):
                    with T.block("B"):
                        i0 = T.axis.spatial(8, i0_0 * 4 + i0_1 * 2 + i0_2)
                        T.reads(A[i0])
                        T.writes(B[i0])
                        B[i0] = A[i0] + T.float32(1)
"""
```

## Relay

**relay.build()**

return

* execution graph in json format
* TVM module library of compiled functions specifically for this graph on the target hardware
* parameter blobs of the model

During the compilation, 

* Relay does the graph-level optimization,
* TVM does the tensor-level optimization,

resulting in an optimized runtime module for model serving.

### Relay v.s. TE

```python
tvm.tir.sqrt(x: tvm.ir.PrimExpr) -> tvm.ir.PrimExpr

# Alias of tvm.tir.sqrt()
tvm.te.sqrt(x: tvm.ir.PrimExpr) -> tvm.ir.PrimExpr

tvm.relay.sqrt(data: tvm.ir.RelayExpr) -> tvm.ir.RelayExpr)
```

For tvm.ir.BaseExpr,

* PrimExpr is class of all primitive expressions, used in the low-level code optimizations and integer analysis.
* RelayExpr is class of all non-primitive expressions.

### Build

`tvm.build` v.s. `tvm.relay.build`

```python
tvm.relay.build(ir_mod: IRModule, target, target_host, executor=graph{}, runtime=cpp) 
    -> tvm.relay.backend.executor_factory.ExecutorFactoryModule

tvm.build(inputs: Union[tvm.te.schedule.Schedule, tvm.tir.function.PrimFunc, tvm.ir.module.IRModule, Mapping[str, tvm.ir.module.IRModule]], args, target, target_host, runtime, binds) 
    -> tvm.driver.build_module.OperatorModule
```

## Optimization

**Blocking**, **Cache**

通过分 Block, 让留在 cache 中的中间计算结果能够发挥作用，从而提升性能。

**Vectorization**, **Array Packing**

连续的内存访问会比较高效，通过 `vectorize` 达到这样的目的。对矩阵的 Array Packing 也是在做这样的优化。

**Parallelization**

非依赖的情况下可以使用并行策略提高整体性能。

## TOPI

```python
n = te.var("n")
m = te.var("m")
A = te.placeholder((n, m), name="A")
```

origin
```python
k = te.reduce_axis((0, m), "k")
B = te.compute((n,), lambda i: te.sum(A[i, k], axis=k), name="B")
s = te.create_schedule(B.op)
# print(tvm.lower(s, [A], simple_mode=True))
```

TOPI 
```python
C = topi.sum(A, axis=1)
ts = te.create_schedule(C.op)
# print(tvm.lower(ts, [A], simple_mode=True))
```

## Develop

* write a pass with IR
* glue to lowering

每一个 phase 做的 transformation 如下，

* Phase 0 generates the raw IR and loop levels.
* Phase 1 flattens the array storage.
* Phase 2 transforms loops, like unroll, vectorization and thread-binding.
* Phase 3 does some cleanup work.

所以比如自定义的 vectorize 适合放在 phase 1 之后。

```python
with tvm.transform.PassContext(config={"tir.add_lower_pass": [(1, vectorize)]}):
    print(tvm.lower(sch, [a, b, c]))
```



## Reference

* [https://tvm.apache.org/docs/](https://tvm.apache.org/docs/)
* [schedule_primitives](https://tvm.apache.org/docs/how_to/work_with_schedules/schedule_primitives.html)
* [optimization TE](https://tvm.apache.org/docs/tutorial/tensor_expr_get_started.html)
* [custom pass](https://tvm.apache.org/docs/how_to/extend_tvm/low_level_custom_pass.html)

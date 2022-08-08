# TVM

## Expression

**TE**
Tensor Expression, DSL (domain-specific language)

**Relay** TVM’s high-level model language

**IR** intermediate representation

**TIR** Tensor Intermediate Representation, TVM’s low-level intermediate representation

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

## Demo

```python
from tvm.driver import tvmc

model = tvmc.load('resnet50-v2-7.onnx', shape_dict={'data':[1, 3, 224, 224]}) #Step 1: Load

tuning_records = tvmc.tune(model, target="llvm") #Step 1.5: Optional Tune

package = tvmc.compile(model, target="llvm", tuning_records = tuning_records) #Step 2: Compile

result = tvmc.run(package, device="cpu") #Step 3: Run

print(result)
```

* IRModule: relay.Function + tir.PrimFunc
* tvmc.compile: relay::Function --> tir::PrimFunc

## Schedule Primitives

**split**

## Reference

* [https://tvm.apache.org/docs/](https://tvm.apache.org/docs/)
* [schedule_primitives](https://tvm.apache.org/docs/how_to/work_with_schedules/schedule_primitives.html)
* [optimization TE](https://tvm.apache.org/docs/tutorial/tensor_expr_get_started.html)

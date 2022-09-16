# Framework

## Program

Python Program 定义
```
# python/paddle/fluid/framework.py
class Program(object):
```

CPP 定义
```
core.ProgramDesc()
# paddle/fluid/framework/program_desc.h
class ProgramDesc {}
```

对应 proto 定义
```
# paddle/fluid/framework/framework.proto
```

全局 default program

```
paddle.static.default_startup_program()
# _main_program_ = Program()
paddle.static.default_main_program()
# _startup_program_ = Program()
```

* startup_program: 模型参数初始化、优化器参数初始化、reader初始化
* main_program: 前向计算、反向计算、模型参数更新、优化器参数更新

CompiledProgram 即 Graph

```
# python/paddle/fluid/compiler.py
class CompiledProgram(object):
```

Program to Graph 
```
core.Graph(program.desc)
```

Graph to Program
```
compiled_program._compile(...)
compiled_graph = compiled_program._graph
fluid.framework.IrGraph(compiled_graph).to_program()
```

## Demo

静态图demo

```python
import paddle
import numpy as np

paddle.enable_static()

inputs = paddle.static.data(name='input', shape=[None, 100], dtype='float32')
outputs = paddle.static.data(name='output', shape=[None, 10], dtype='float32')

out = paddle.static.nn.fc(x=inputs, size=10, activation='relu')

cost = paddle.nn.functional.square_error_cost(input=out, label=outputs)
loss = paddle.mean(cost)
adam = paddle.optimizer.Adam(learning_rate=1e-3)
adam.minimize(loss)

startup_program = paddle.static.default_startup_program()
main_program = paddle.static.default_main_program()

op_types = [op.type for op in startup_program.global_block().ops]
print(op_types)

op_types = [op.type for op in main_program.global_block().ops]
print(op_types)

executor = paddle.static.Executor()
executor.run(startup_program)

compiled_program = paddle.static.CompiledProgram(main_program)

BATCH_NUM = 20
BATCH_SIZE = 32

for batch_id in range(BATCH_NUM):
    input_data = np.random.random([BATCH_SIZE, 100]).astype('float32')
    output_data = np.random.random([BATCH_SIZE, 10]).astype('float32')
    loss_numpy, = executor.run(main_program, feed={'input': input_data, 'output': output_data}, fetch_list=[loss])
    print("Batch {}, loss = {}".format(batch_id, loss_numpy))
```

startup_program 的 op

```
['uniform_random', 'fill_constant', 'fill_constant', 'fill_constant', 'fill_constant', 'fill_constant', 'fill_constant', 'fill_constant', 'fill_constant', 'fill_constant', 'fill_constant']
```

main_program 的 op

```
['mul', 'elementwise_add', 'relu', 'elementwise_sub', 'square', 'reduce_mean', 'fill_constant', 'reduce_mean_grad', 'square_grad', 'elementwise_sub_grad', 'relu_grad', 'elementwise_add_grad', 'mul_grad', 'adam', 'adam']
```

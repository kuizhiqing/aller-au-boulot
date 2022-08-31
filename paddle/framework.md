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

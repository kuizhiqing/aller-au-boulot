# CINN

## Summary

**PassManger**

```python
core.ProgramDesc(main_program.desc)
core.apply_pass(tmp_main_program, ...)
main_program._rebuild_from_desc(tmp_main_program)

# apply_pass
framework::ir::Pass::ApplyPassesToProgram(main_program, ...)
Graph graph(*main_program)
pass->Apply(&graph);
ConvertToPrograms(&graph, main_program, ...);
```

**Excutor build**

```cpp
graph = core.Graph(program.desc)
graph = pass->Apply(graph)
ir_graph = fluid.framework.IrGraph(graph)
ir_graph.to_program()
```

## PassManager

Usage

```python
from paddle.distributed.passes import new_pass, PassManager
pass_manager = PassManager([
    new_pass("build_cinn"),
    new_pass("fuse_elewise_add_act"),
])
pass_manager.apply([main_prog], [startup_prog])
op_types = [op.type for op in main_prog.global_block().ops]
self.assertTrue('cinn_launch' in op_types)
```

```python
# python/paddle/fluid/framework.py

from paddle.fluid.framework import core, _apply_pass

def _apply_pass(main_program,
                startup_program,
                pass_name,
                pass_attrs={},
                pass_attr_types={}):
    assert isinstance(pass_attrs, dict), "pass_attrs must be dict"
    assert isinstance(pass_attr_types, dict), "pass_attr_types must be dict"
    tmp_main_program = core.ProgramDesc(main_program.desc)
    tmp_startup_program = core.ProgramDesc(startup_program.desc)
    attrs = core.apply_pass(tmp_main_program, tmp_startup_program, pass_name,
                            pass_attrs, pass_attr_types)
    main_program._rebuild_from_desc(tmp_main_program)
    startup_program._rebuild_from_desc(tmp_startup_program)
    return attrs
```

```cpp
// paddle/fluid/pybind/ir.cc

m->def("apply_pass",
         [](framework::ProgramDesc *main_program,
            framework::ProgramDesc *startup_program,
            const py::object &py_pass_names,
            const std::unordered_map<std::string, py::object> &pass_attrs,
            std::unordered_map<std::string, std::string> pass_attr_types) {
           auto pass_names = GetPassNames(py_pass_names);
           std::vector<std::unique_ptr<framework::ir::Pass>> passes;
           std::vector<const framework::ir::Pass *> passes_not_owned;
           passes.reserve(pass_names.size());
           passes_not_owned.reserve(pass_names.size());
           for (const auto &name : pass_names) {
             auto pass = framework::ir::PassRegistry::Instance().Get(name);
             SetAttrsToPass(pass_attrs, &pass_attr_types, pass.get());
             passes.push_back(std::move(pass));
             passes_not_owned.push_back(passes.back().get());
           }

           framework::ir::Pass::ApplyPassesToProgram(
               passes_not_owned, main_program, startup_program);
           std::unordered_map<std::string, py::object> result_attrs;
           for (const auto &pass : passes) {
             for (const auto &name_and_value : pass_attrs) {
               const auto &attr_name = name_and_value.first;
               const auto &attr_type = pass_attr_types.at(attr_name);
               result_attrs[attr_name] =
                   PassAttrGetterSetterRegistry::Instance().Get(
                       *pass, attr_name, attr_type);
             }
           }
           return result_attrs;
         });
```

```cpp
// paddle/fluid/framework/ir/pass.cc

void Pass::ApplyPassesToProgram(const std::vector<const Pass *> &passes,
                                ProgramDesc *main_program,
                                ProgramDesc *startup_program) {
 if (passes.size() == 1 && !passes[0]->SupportApplyProgramViaGraph()) {
    // apply pass to program
    passes[0]->ApplyImpl(main_program, startup_program);
    FillNotSpecifiedOpRole(*main_program);
    return;
  }

  Graph graph(*main_program);
  for (auto *p : passes) {
    p->Apply(&graph);
  }
  ConvertToPrograms(&graph, main_program, startup_program);
  FillNotSpecifiedOpRole(*main_program);
}

Graph *Pass::Apply(Graph *graph) const {
    ApplyImpl(graph);
    return graph;
}
```

## Excutor Apply

```python
def _compile(program, loss_name=None):
    build_strategy = paddle.static.BuildStrategy()
    exec_strategy = paddle.static.ExecutionStrategy()

    exec_strategy.num_threads = 1

    compiled_program = paddle.static.CompiledProgram(
        program).with_data_parallel(
            loss_name=loss_name,
            build_strategy=build_strategy,
            exec_strategy=exec_strategy)

    return compiled_program

executor = paddle.static.Executor()

compiled_program = _compile(program_with_fetch_op, loss_name)

compiled_program._compile(scope, paddle.framework._current_expected_place())
compiled_graph = compiled_program._graph
ir_graph = fluid.framework.IrGraph(compiled_graph, for_test=True)
ir_program = ir_graph.to_program()
```

```python
# python/paddle/fluid/compiler.py

BuildStrategy = core.ParallelExecutor.BuildStrategy

class CompiledProgram(object):
    # Static Graph
    def __init__(self, program_or_graph, build_strategy=None):
        self._graph = core.Graph(program_or_graph.desc)
        self._program = program_or_graph
    def _compile(self, scope, place):
        self._executor = self._compile_data_parallel(...)
    def _compile_data_parallel(self, places, use_device, scope=None):
        self._build_strategy = BuildStrategy()
        core.ParallelExecutor(...)

```

```cpp
// paddle/fluid/pybind/parallel_executor.cc

py::class_<ParallelExecutor> pe(m, "ParallelExecutor");
py::class_<BuildStrategy> build_strategy(pe, "BuildStrategy", R"DOC(
```

```cpp
// paddle/fluid/framework/parallel_executor.cc

ParallelExecutor::ParallelExecutor(const std::vector<platform::Place> &places,
                                   const std::vector<std::string> &bcast_vars,
                                   const std::string &loss_var_name,
                                   Scope *scope,
                                   const std::vector<Scope *> &local_scopes,
                                   const ExecutionStrategy &exec_strategy,
                                   const BuildStrategy &build_strategy,
                                   ir::Graph *graph){
    // ParallelExecutorPrivate *member_;
    std::vector<ir::Graph *> async_graphs = CompileGraphWithBuildStrategy(graph, &graphs, loss_var_name);
    graph = member_->ApplyMemoryOptimizePass(graph);
    std::vector<ir::Graph *> final_graphs = CreateSSAGraphExecutor(exec_strategy, &async_graphs, graph);
    if (!member_->build_strategy_.async_mode_) {
      member_->executor_.reset(new details::ScopeBufferedSSAGraphExecutor(
        exec_strategy,
        member_->local_scopes_,
        member_->local_exec_scopes_,
        std::move(var_infos),
        member_->places_,
        std::move(member_->executor_)));
  }
}

std::vector<ir::Graph *> ParallelExecutor::CompileGraphWithBuildStrategy(
    ir::Graph *graph,
    std::vector<ir::Graph *> *device_graphs,
    const std::string &loss_var_name) {
    graph = member_->build_strategy_.Apply(graph, ...);
}
```

```cpp
// paddle/fluid/framework/details/build_strategy.cc

ir::Graph *BuildStrategy::Apply(ir::Graph *graph, ...){
    // 这里使用 ParallelExecutorPassBuilder 添加 pass
    CreatePassesFromStrategy(false);
    for (std::shared_ptr<ir::Pass> &pass : pass_builder_->AllPasses()) {
        if (FLAGS_convert_all_blocks) {
          for (size_t i = 0; i < graph->SubGraphsSize(); ++i) {
            pass->Apply(graph->GetSubGraph(i));
          }
        } else {
          graph = pass->Apply(graph);
        }
    }
}

std::shared_ptr<ir::PassBuilder> BuildStrategy::CreatePassesFromStrategy(bool finalize_strategy) const {
    pass_builder_.reset(new ParallelExecutorPassBuilder(*this));
    return pass_builder_;
}

class ParallelExecutorPassBuilder : public ir::PassBuilder {
    ...
    AppendPass("build_cinn_pass");
}
```

## Build CINN Pass

cinn pass 通过转成 Graph 然后 apply

```cpp
// paddle/fluid/framework/paddle2cinn/build_cinn_pass.cc

void BuildCinnPass::ApplyImpl(Graph* graph) const { SearchAllSubgraphs(graph); }

void SearchAllSubgraphs(Graph* graph) {
    std::vector<GraphNodeVec> clusters = framework::ir::SubgraphDetector(graph, teller)();
    for (const auto& node_vec : clusters) {
        cinn_compiler->AddGraph(CreateNewSubGraph(...)
        ReplaceSubGraphWithCinnOpNode(...)
    }
}

void ReplaceSubGraphWithCinnOpNode(...){
    // Add the cinn op node whose name is "kCinnLaunchOp" into graph
    AddCinnOpToGraph(...);
    // Remove the cinn subgraph from graph
    RemoveSubGraphFromGraph(cluster, cluster_internals, graph);
}
```

```cpp
// paddle/fluid/framework/ir/subgraph_detector.cc

std::vector<std::vector<Node *>> SubgraphDetector::operator()() {
  MarkNodesInsideSubGraph();
  return ExtractSubGraphs();
}

void SubgraphDetector::MarkNodesInsideSubGraph() {
  for (auto &node : framework::ir::GraphTraits::DFS(*graph_)) {
    if (node_inside_subgraph_teller_(&node)) {
      Agent(&node).set_marked(true);
      if (node.IsOp()) {
        // If a function is inside the sub-graph, mark all the output variables
        // to be inside too, so that two marked functions will be inside a same
        // sub-graph, lets take a example:  A_function->var->B_function, if
        // A_function is marked, var should also be marked, so that B_function
        // will be in the same sub-graph with A_function if B_function is
        // marked.
        MarkOutLinksInSubGraph(&node);
      }
    }
  }
}

std::vector<std::vector<Node *>> SubgraphDetector::ExtractSubGraphs() {
}
```

## Prim op

通过以下API操作全局变量使用

```
paddle.incubate.autograd.enable_prim()
paddle.incubate.autograd.disable_prim()
paddle.incubate.autograd.prim_enabled()
```

具体影响由使用 AD API 时体现

```
# python/paddle/incubate/autograd/primapi.py
paddle.incubate.autograd.grad()
# 调用 primx.orig2prim(block)
```


# PyTorch Profiler

## Demo

体现基本流程的示例
```python
import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity

# 创建模型，需要 profile 的对象
model = models.resnet18()
inputs = torch.randn(5, 3, 224, 224)

# 配置
prof = profile(activities=[ProfilerActivity.CPU], record_shapes=True)

prof.start()

model(inputs)

prof.stop()

# 结果分析和输出
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

prof.export_chrome_trace("trace.json")
```

其中 `torch.profiler.profile` 可以使用 `with` 语法
```python
with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    with record_function("model_inference"):
        model(inputs)
```

输出
```shell
---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------
                             Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls
---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------
                 model_inferencex         0.10%      15.353ms       100.00%       14.951s       14.951s             1
                 aten::batch_norm         0.00%     292.000us        43.14%        6.449s     322.464ms            20
     aten::_batch_norm_impl_index         0.00%     567.000us        43.13%        6.449s     322.450ms            20
          aten::native_batch_norm        32.92%        4.921s        43.13%        6.448s     322.419ms            20
                     aten::conv2d         0.00%     310.000us        42.00%        6.279s     313.938ms            20
                aten::convolution         0.00%     350.000us        41.99%        6.278s     313.923ms            20
               aten::_convolution         0.00%     601.000us        41.99%        6.278s     313.905ms            20
         aten::mkldnn_convolution        41.98%        6.276s        41.99%        6.278s     313.875ms            20
                       aten::mean         0.01%       1.209ms        10.54%        1.576s      75.043ms            21
                        aten::sum        10.45%        1.562s        10.45%        1.562s      74.386ms            21
---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 14.951s
```

### API

`torch.profiler.profile`

参数说明
* `activities` list 类型，profile 的内容，支持 `torch.profiler.ProfilerActivity.CPU` 和 `torch.profiler.ProfilerActivity.CUDA`，这里的设置需要和模型使用的 device 一致
* `schedule` 默认会持续记录所有事件，使用 scheduler() 作为帮助函数生成 schedule 函数，自定义记录逻辑
* `on_trace_ready` 配合 `schedule` 使用，在它返回 `ProfilerAction.RECORD_AND_SAVE` 后被调用
* `record_shapes` 是否记录 input shapes
* `profile_memory` 是否记录 内存/显存, 和 `activities` 对应
* `with_stack` 是否开启调用文件信息源的记录，包括代码文件和行号
* `with_flops` 预估FLOPs，主要针对 matrix multiplication and 2D convolution
* `with_modules` 层级记录，暂时只针对 TorchScript

`ProfilerAction` 用于状态的记录和转换
```python
class ProfilerAction(Enum):
    NONE = 0
    WARMUP = 1
    RECORD = 2
    RECORD_AND_SAVE = 3
```

`profile` 对象

```python
# torch/profiler/profiler.py

# Profiler context manager
class profile(_KinetoProfile):

    def __init__(...):
        # 记录函数
        self.step_rec_fn: Optional[prof.record_function] = None

        # 状态转换时会触发一系列操作，action_map 记录里任意两个状态转换时执行的动作
        self.action_map: Dict[Tuple[ProfilerAction, Optional[ProfilerAction]], List[Any]] = {
            (ProfilerAction.NONE, ProfilerAction.WARMUP): [self.prepare_trace],
            (ProfilerAction.NONE, ProfilerAction.RECORD): [self.prepare_trace, self.start_trace],
            ...
        }

    def start(self):
        self._transit_action(ProfilerAction.NONE, self.current_action)
        if self.record_steps:
            self.step_rec_fn = prof.record_function("ProfilerStep#" + str(self.step_num))
            self.step_rec_fn.__enter__()

    def stop(self):
        if self.record_steps and self.step_rec_fn:
            self.step_rec_fn.__exit__(None, None, None)
        self._transit_action(self.current_action, None)

    def step(self):
        self.step_num += 1
        # schedule 接受 step 数，返回当前 action
        self.current_action = self.schedule(self.step_num)

        # 转换状态，触发 map 中定义的动作
        self._transit_action(prev_action, self.current_action)

        if self.record_steps:
            self.step_rec_fn = prof.record_function("ProfilerStep#" + str(cur_step))
            self.step_rec_fn.__enter__()
```

`schedule`

```python
def schedule(*, wait: int, warmup: int, active: int, repeat: int = 0, skip_first: int = 0) -> Callable:
    # skip_fist + ( wait + warmup + active ) * repeat
    # NONE                 WARMUP   RECORD RECORD_AND_SAVE
    def schedule_fn(step: int) -> ProfilerAction:
        # 根据 step 返回 当前的状态
    return schedule_fn
```

`record_function`

```python
# torch/autograd/profiler.py

class record_function(ContextDecorator):
    def __init__(self, name: str, args: Optional[str] = None):
        self.record = torch.jit.annotate(Optional["torch.classes.profiler._RecordFunction"], None)

    def __enter__(self):
        self.record = torch.ops.profiler._record_function_enter_new(self.name, self.args)

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any):
        torch.ops.profiler._record_function_exit(self.record)
```



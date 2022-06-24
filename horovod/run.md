# Run

### 启动
```python
setup(name='horovod',
      entry_points={
          'console_scripts': [
              'horovodrun = horovod.runner.launch:run_commandline'
          ]
      })
```


```python
# horovod/runner/launch.py

def run_commandline():
    args = parse_args()
    _run(args)

def _run(args):
    # set args.hosts
    if _is_elastic(args):
        return _run_elastic(args)
    else:
        return _run_static(args)
```

### 非弹性启动
```python
def _run_static(args):
    settings = hvd_settings.Settings(...)
    nics = driver_service.get_common_interfaces(settings, all_host_names,
                                                remote_host_names, fn_cache)
    if args.run_func:
        executable = args.executable or sys.executable
        command = [executable, '-m', 'horovod.runner.run_task', str(driver_ip), str(run_func_server_port)]
    else:
        command = args.command
    _launch_job(args, settings, nics, command)

def _launch_job(args, settings, nics, command):
    def gloo_run_fn():
        driver_ip = network.get_driver_ip(nics)
        gloo_run(settings, nics, env, driver_ip, command)

    def mpi_run_fn():
        mpi_run(settings, nics, env, command)

    def js_run_fn():
        js_run(settings, nics, env, command)

    run_controller(args.use_gloo, gloo_run_fn,
                   args.use_mpi, mpi_run_fn,
                   args.use_jsrun, js_run_fn,
                   args.verbose)

def run_controller(use_gloo, gloo_run, use_mpi, mpi_run, use_jsrun, js_run, verbosity):
    if use_gloo:
        gloo_run()
    elif use_mpi:
        mpi_run()
    elif use_jsrun:
        js_run()

from horovod.runner.gloo_run import gloo_run, gloo_run_elastic
from horovod.runner.mpi_run import mpi_run
from horovod.runner.js_run import js_run, is_jsrun_installed
```

```python
# horovod/runner/gloo_run.py

def gloo_run(settings, nics, env, server_ip, command):
    # 启动命令通过 ssh 分发，如果出错所有进程将被 kill
    # 先封装执行函数
    exec_command = _exec_command_fn(settings)
    # 再调用执行
    launch_gloo(command, exec_command, settings, nics, env, server_ip)

def _exec_command_fn(settings):
    def _exec_command(command, slot_info, events):
        # 如果是需要分发到 remote 的节点
        # from horovod.runner.util.remote import get_remote_command
        # get_remote_command 提供 ssh 封装
        if host_address not in local_addresses:
            command = get_remote_command(local_command,...)
        exit_code = safe_shell_exec.execute(command,...)
    return _exec_command

def launch_gloo(command, exec_command, settings, nics, env, server_ip):
    # exec_command 为执行的命令
    # args_list 是执行的参数，由每个节点所需参数组成的列表
    # 通过如下方法的调用实现多节点运行
    res = threads.execute_function_multithreaded(exec_command, args_list, block_until_all_done=True)
```

```python
# horovod/runner/util/threads.py

def execute_function_multithreaded(fn, args_list, block_until_all_done=True, max_concurrent_executions=1000):
    worker_queue = queue.Queue()
    result_queue = queue.Queue() # 结果池，用于放置结果，后续忽略

    # 把任务放进任务池
    for i, arg in enumerate(args_list):
        worker_queue.put(arg)

    # 只要任务池里还有任务就取出来执行之
    def fn_execute():
        while True:
            arg = worker_queue.get(block=False)
            exec_index = arg[-1]
            res = fn(*arg[:-1])

    # 启动多线程分发命令，感觉必要性不大
    for _ in range(number_of_threads):
        thread = in_thread(target=fn_execute, daemon=not block_until_all_done)

def in_thread(target, args=(), name=None, daemon=True, silent=False):
    bg = threading.Thread(target=fn, args=args, name=name)
    bg.daemon = daemon
    bg.start()
```

```python
# horovod/runner/common/util/safe_shell_exec.py

# 使用 multiprocessing.Process 启动进程
# 然后再使用 subprocess 启动进程执行

def execute(command, env=None, stdout=None, stderr=None, index=None, events=None,
            prefix_output_with_timestamp=False):
    ctx = multiprocessing.get_context('spawn')

    exit_event = _create_event(ctx)

    # 当 parent process 被 hard kill 时，这个 Pipe 会被关闭，然后 middleman 就会向子进程发送 SIGTERM，避免出现 orphaned process
    (r, w) = ctx.Pipe(duplex=False)

    middleman = ctx.Process(target=_exec_middleman, args=(command, env, exit_event, ..., (r, w)))
    middleman.start()

    middleman.join()
    return middleman.exitcode

def _exec_middleman(command, env, exit_event, stdout, stderr, rw):
    os.setsid()

    executor_shell = subprocess.Popen(command, shell=True, env=env,
                                      stdout=stdout_w, stderr=stderr_w)

```

```python
# horovod/runner/mpi_run.py

def mpi_run(settings, nics, env, command, stdout=None, stderr=None):
    mpirun_command = (
        'mpirun {basic_args} '
        '-np {num_proc}{ppn_arg}{hosts_arg} '
        '{binding_args} '
        '{mpi_args} '
        '{mpi_ssh_args} '
        '{tcp_intf_arg} '
        '{nccl_socket_intf_arg} '
        '{output_filename_arg} '
        '{env} {extra_mpi_args} {command}'
    )

    if settings.run_func_mode:
        exit_code = safe_shell_exec.execute(mpirun_command, env=env, stdout=stdout, stderr=stderr)
    else:
        os.execve('/bin/sh', ['/bin/sh', '-c', mpirun_command], env)
```

### 弹性启动
```python
def _run_elastic(args):
    settings = elastic_settings.ElasticSettings(discovery=discover_hosts,...)
    return gloo_run_elastic(settings, env, args.run_func if args.run_func else args.command, executable)

from horovod.runner.gloo_run import gloo_run, gloo_run_elastic
```

```python
```

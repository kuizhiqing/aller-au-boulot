# Concurrent Execution

**TL;DR;**

* 使用 python 作为引子/driver 启动其他独立进程，使用 exec. 
* 使用 python 作为主进程管理，子进程是独立启动的进程，使用 subprocess.Popen.
* 子进程是 python 函数使用 multiprocessing。
* 把 python 函数当作线程启动时使用 threading。

[https://docs.python.org/3/library/concurrency.html](https://docs.python.org/3/library/concurrency.html)

### 1. subprocess

* 启动子进程，自定义 excutable

```python
import subprocess

proc = subprocess.Popen(["/usr/bin/git", "commit", "-m", "Fixes a bug."])
proc.poll()
proc.wait()
proc.communicate()
proc.terminate()
proc.kill()
```
[https://docs.python.org/3/library/subprocess.html](https://docs.python.org/3/library/subprocess.html)

### 2. threading

* 启动线程
* 存在 GIL 问题，无法完全利用 CPU

```python
import threading

def serve():
    pass

thread = threading.Thread(target=serve, daemon=None)
thread.start()
thread.join()
```

[https://docs.python.org/3/library/threading.html](https://docs.python.org/3/library/threading.html)

#### Pros
* Lightweight - low memory footprint
* Shared memory - makes access to state from another context easier
* Allows you to easily make responsive UIs
* cPython C extension modules that properly release the GIL will run in parallel
* Great option for I/O-bound applications

#### Cons

* cPython - subject to the GIL
* Not interruptible/killable
* If not following a command queue/message pump model (using the Queue module), then manual use of synchronization primitives become a necessity (decisions are needed for the granularity of locking)
* Code is usually harder to understand and to get right - the potential for race conditions increases dramatically


### 3. multiprocessing

* 启动子进程，使用内置函数
* 使用进程，充分利用 CPU 资源

```python
from multiprocessing

proc = multiprocessing.Process(target=serve, daemon=None)
proc.start()
proc.join()
proc.close()

```
[https://docs.python.org/3/library/multiprocessing.html](https://docs.python.org/3/library/multiprocessing.html)


#### Pros
* Separate memory space
* Code is usually straightforward
* Takes advantage of multiple CPUs & cores
* Avoids GIL limitations for cPython
* Eliminates most needs for synchronization primitives unless if you use shared memory (instead, it's more of a communication model for IPC)
* Child processes are interruptible/killable
* Python multiprocessing module includes useful abstractions with an interface much like threading.Thread
* A must with cPython for CPU-bound processing

#### Cons

* IPC a little more complicated with more overhead (communication model vs. shared memory/objects)
* Larger memory footprint

### 4. Executor

[https://docs.python.org/3/library/concurrent.futures.html](https://docs.python.org/3/library/concurrent.futures.html)

* ProcessPoolExecutor 封装 multiprocessing module，提供进程池
* ThreadPoolExecutor, 线程池

API
```
class concurrent.futures.Executor:
    submit(fn, /, *args, **kwargs)
    map(func, *iterables, timeout=None, chunksize=1)
    shutdown(wait=True, *, cancel_futures=False)
```

调用 submit 返回 `concurrent.futures.Future`, 使用 result(timeout=None) 获取结果。.

### exec

python 转 shell 执行

```
os.execve('/bin/sh', 'echo', 'ok')
```

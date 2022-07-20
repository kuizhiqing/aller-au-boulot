# Ray

## Reference

* [Ray Github](https://github.com/ray-project/ray)
* [Ray Documentation](https://docs.ray.io/en/latest/)
* [Ray Tips](https://rise.cs.berkeley.edu/blog/ray-tips-for-first-time-users/)
* [Ray 1.x Architecture - Google 文档](https://docs.google.com/document/d/1lAy0Owi-vPz2jEqBSaHNQcy2IBSDEHyXNOQZlGuj93c/preview#)


源代码和官方文档永远是最好的学习资料，总结和学习笔记能辅助快速理解，抓住重点，提高效率。


| **API**     | **Description**                                                                                                                                                                                          | **Example**                                                                                                                  |
| ----------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| ray.init()  | Initialize Ray context.                                                                                                                                                                                  |                                                                                                                              |
| @ray.remote | Function or class decorator specifying that the function will be executed as a task or the class as an actor in a different process.                                                                     | @ray.remote @ray.remote <br> def fun(x):           class Actor(object):<br> …                            def method(y)<br> … |
| .remote     | Postfix to every remote function, remote class declaration, or invocation of a remote class method. Remote operations are *asynchronous*.                                                                | ret_id = fun.remote(x)<br>a = Actor.remote()<br>ret_id = a.method.remote(y)                                                  |
| ray.put()   | Store object in object store, and return its ID. This ID can be used to pass object as an argument to any remote function or method call. This is a *synchronous* operation.                             | x_id = ray.put(x)                                                                                                            |
| ray.get()   | Return an object or list of objects from the object ID or list of object IDs. This is a *synchronous* (i.e., blocking) operation.                                                                        | x = ray.get(x_id)<br>…<br>objects = ray.get(object_ids)                                                                      |
| ray.wait()  | From a list of object IDs returns (1) the list of IDs of the objects that are ready, and (2) the list of IDs of the objects that are not ready yet. By default it returns one ready object ID at a time. | ready_ids, not_ready_ids =  ray.wait(object_ids)                                                                             |

特点

- 分布式异步调用
- 内存调度
- Pandas/Numpy 的分布式支持
- 支持 Python
- 整体性能出众

Ray (pickle5 + cloudpickle)

[GitHub - ray-project/plasma: A minimal shared memory object store design](https://github.com/ray-project/plasma)

[GitHub - cloudpipe/cloudpickle: Extended pickling support for Python objects](https://github.com/cloudpipe/cloudpickle)

* Plasma is an in-memory object store that is being developed as part of Apache Arrow
* Ray uses Plasma to efficiently transfer objects across different processes and *different nodes*

#### Actor

* An actor is essentially a stateful worker(or a service)
* A new actor is instantiated, a new worker is created
* When an actor contains async methods, the actor will be converted to async actors. This means all the ray’s tasks will run as a coroutine.

```python
## 1
@ray.remote
class Counter(object):
    def __init__(self):
        self.value = 0
Counter = ray.remote(Counter)

## 1
class Counter(object):
    def __init__(self):
        self.value = 0

## then
counter_actor = Counter.remote()
```

#### Actor vs Worker

Actor: a worker instantiated at runtime

Worker: python process, execute multiple tasks or actor (dedicated)

#### Fault Tolerance

* task retry
* actor retry
* object retrieve or reconstruction

#### Airflow

Airflow can be act as job manager in Ray


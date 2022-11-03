# decorator

## Examples

```python
def decorator(f):
    print(0)
    return f

@decorator
def func(*args, **kwargs):
    print(2, *args, **kwargs)
    return 3

# func = decorator(func)

print(1)
print(func())
```

应用场景

* 插件注册 Registering Plugins
* Trace


```python
def decorator(f):
    print(0)
    def wrapper(*args, **kwargs):
        print(2)
        ret = f(*args, **kwargs)
        print(4)
        return ret
    return wrapper

@decorator
def f(*args, **kwargs):
    print(3)
    return 5

# f = decorator(f)

print(1)
print(f())
```

应用场景

* 过滤/校验
* 重复调用
* 统计时间
* Debug


```python
def decorator(arg):
    print(arg)
    def wrapper(f):
        print(1)
        return f
    return wrapper

@decorator(0)
def f(*args, **kwargs):
    print(3)
    return 4

# f = decorator(arg)(f)

print(2)
print(f())
```

```python
import functools

def repeat(_func=None, *, num_times=2):
    def decorator_repeat(func):
        @functools.wraps(func)
        def wrapper_repeat(*args, **kwargs):
            for _ in range(num_times):
                value = func(*args, **kwargs)
            return value
        return wrapper_repeat

    if _func is None:
        return decorator_repeat
    else:
        return decorator_repeat(_func)

@repeat
def say_whee():
    print("Whee!")

@repeat(num_times=3)
def greet(name):
    print(f"Hello {name}")

say_whee()
greet("kk")
```

```python
import functools

def singleton(cls):
    """Make a class a Singleton class (only one instance)"""
    @functools.wraps(cls)
    def wrapper_singleton(*args, **kwargs):
        if not wrapper_singleton.instance:
            wrapper_singleton.instance = cls(*args, **kwargs)
        return wrapper_singleton.instance
    wrapper_singleton.instance = None
    return wrapper_singleton

@singleton
class TheOne:
    pass
```

```python
class Decorator:
    def __init__(self, arg):
        print(arg)
        self.arg = arg

    def __call__(self, f):
        print(1, self.arg)
        return f

@Decorator(0)
def func(arg):
    print(3)
    return arg

# func = Decorator(arg)(arg)

print(2)

print(func(4))
print(func(5))
```

```python
class Decorator:
    def __init__(self, f):
        print(0)
        self.f = f

    def __call__(self, arg):
        print(2, self.f)
        return self.f(arg)

@Decorator
def func(arg):
    print(3)
    return arg

# func = Decorator(arg)(arg)

print(1)

print(func(4))
print(func(5))
```

## Reference

* [realpython](https://realpython.com/primer-on-python-decorators/)

"""

第51条 优先考虑通过类装饰器来提供可组合的扩充功能， 不要使用元类

"""

from functools import wraps


def trace_func(func):
    if hasattr(func, 'tracing'):
        return func

    @wraps(func)
    def wrapper(*args, **kwargs):
        result = None
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            result = e
            raise
        finally:
            print(f'{func.__name__}({args!r},{kwargs!r})->{result!r}')

    wrapper.tracing = True
    return wrapper


class TraceDict(dict):
    @trace_func
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @trace_func
    def __setitem__(self, *args, **kwargs):
        super().__setitem__(args, kwargs)

    @trace_func
    def __getitem__(self, *args, **kwargs):
        super().__getitem__(*args, **kwargs)


trace_dict = TraceDict({'hi': 1})
trace_dict['there'] = 33
print(trace_dict['hi'])
print(trace_dict)
try:
    trace_dict['does not exist']
except KeyError:
    pass


import types


trace_types=(
    types.MethodType,
    types.FunctionType,
    types.BuiltinFunctionType,
    types.BuiltinMethodType,
    types.MethodDescriptorType,
    types.ClassMethodDescriptorType
)


class TraceMeta(type):
    def __new__(meta, name ,bases, class_dict):
        klass = super().__new__(meta, name, bases, class_dict)
        for key in dir(klass):
            value = getattr(klass, key)
            if isinstance(value, trace_types):
                wrapped = trace_func(value)
                setattr(klass, key, wrapped)

        return klass


class TraceMetaDict(dict, metaclass=TraceMeta):
    pass


trace_meta_dict = TraceMetaDict({'hi': 1})
trace_meta_dict['there'] = 666
print(trace_meta_dict['hi'])


def my_class_decorator(klass):
    klass.extra_param = 'hello'
    return klass


@my_class_decorator
class MyClass:
    pass


print(MyClass)
print(MyClass.extra_param)


def trace(klass):
    for key in dir(klass):
        value = getattr(klass, key)
        if isinstance(value , trace_types):
            wrapped = trace_func(value)
            setattr(klass, key, wrapped)
    return klass


@trace
class TraceDecoratorDict(dict):
    pass


print('[TraceDecoratorDict]+++++++++++++++++++++++++++++++==================')

trace_meta_dict = TraceDecoratorDict({'hi': 1})
trace_meta_dict['there'] = 666
print(trace_meta_dict['hi'])




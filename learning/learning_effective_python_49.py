"""

第49条 用__init_subclass__记录现有子类

"""

import json


class Serializable:
    def __init__(self, *args):
        self.args = args

    def serialize(self):
        return json.dumps({'args':self.args})


class Point2D(Serializable):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.x = x
        self.y = y

    def __repr__(self):
        return f'Point2D({self.x}, {self.y})'


# point = Point2D(5,3)
# print('Object:', point)
# print('Serialized:', point.serialize())


class Deserializable(Serializable):
    @classmethod
    def deserialize(cls, json_data):
        params = json.loads(json_data)
        return cls(*params)


class BetterPoint2D(Deserializable):
    ...


before = BetterPoint2D(4, 5)
print('Before:', before)
data = before.serialize()
print('Serialized:', data)
after = BetterPoint2D.deserialize(data)
print('After:', after)


class BetterSerializable:
    def __init__(self, *args):
        self.args = args

    def serialize(self):
        return json.dumps({
            'class': self.__class__.__name__,
            'args': self.args
        })

    def __repr__(self):
        name = self.__class__.__name__
        args_str = ', '.join(str(x) for x in self.args)
        return f'{name}({args_str})'


registry = {}


def register_class(target_class):
    registry[target_class.__name__] = target_class


def deserialize(data):
    params = json.loads(data)
    name = params['class']
    target_class = registry[name]
    return target_class(*params['args'])


class EventBetterPoint2D(BetterSerializable):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.x = x
        self.y = y


register_class(EventBetterPoint2D)

before = EventBetterPoint2D(2,3)
print('Before:', before)
data = before.serialize()
print('Serialized:', data)
after = deserialize(data)
print('After:', after)


class Meta(type):
    def __new__(meta, name, bases, class_dict):
        cls = type.__new__(meta, name, bases, class_dict)
        register_class(cls)
        return cls


class RegisteredSerializable(BetterSerializable, metaclass=Meta):
    pass


class Vector3D(RegisteredSerializable):
    def __init__(self, x, y, z):
        super().__init__(x, y, z)
        self.x, self.y, self.z = x, y, z


print('[Vector3d]====================================================================')
before = Vector3D(5,6,7)
print('Before', before)
data = before.serialize()
print('Serialized:', data)
print('After:', deserialize(data))


print('[BetterRegisteredSerializable] by init subclass method=======================')
class BetterRegisteredSerializable(BetterSerializable):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__()
        register_class(cls)


class Vector1D(BetterRegisteredSerializable):
    def __init__(self, magnitude):
        super().__init__(magnitude)
        self.magnitude = magnitude


before = Vector1D(2222)
print('Before:', before)
data = before.serialize()
print('Serialized:', data)
print("after:", deserialize(data))






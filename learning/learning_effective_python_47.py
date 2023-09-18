""""

第47条 针对惰性属性使用__getattr__, __getattribute__ 及 __setattr__

"""


class LazyRecord:
    def __init__(self):
        self.exists = 5

    def __getattr__(self, name):
        value = f'value for {name}'
        setattr(self, name, value)
        return value


# data = LazyRecord()
# print('Before:', data.__dict__)
# print('foo:', data.foo)
# print('After:', data.__dict__)


class LoggingLazyRecord(LazyRecord):
    def __getattr__(self, name):
        print(f'* Called __getattr__({name!r}),', f'populating instance dictionary', self.__dict__)
        result = super().__getattr__(name)
        print(f'* Returning {result!r}')
        return result


# data = LoggingLazyRecord()
# print('exists:', data.exists)
# print('First foo:', data.foo, data.__dict__)
# print('Second foo:', data.foo, data.__dict__)

class ValidatingRecord:
    def __init__(self):
        self.exists = 5

    def __getattribute__(self, name):
        print(f'* Called __getattribute__({name})')
        try:
            value = super().__getattribute__(name)
            print(f'* Found {name!r}, returning {value!r}')
            return value
        except AttributeError:
            value = f'value for {name}'
            print(f'* Setting {name!r} to {value!r}')
            setattr(self, name,value)
            return value


# data = ValidatingRecord()
# print('exists:', data.exists)
# print('First foo:', data.foo)
# print('Second foo:', data.foo)


class MissingPropertyRecord:
    def __getattr__(self, name):
        if name == 'bad_name':
            raise AttributeError(f'{name} is missing')


# missing_data = MissingPropertyRecord()
# print(missing_data.bad_name)

# data = LoggingLazyRecord()
# print('Before:', data.__dict__)
# print('Has first foo:', hasattr(data, 'foo'))
# print('After:', data.__dict__)
# print('Has second foo:', hasattr(data, 'foo'))

# data = ValidatingRecord()
# print('Has first foo:', hasattr(data, 'foo'))
# print('Has second foo:', hasattr(data, 'foo'))


class SavingRecord:
    def __setattr__(self, name, value):
        super().__setattr__(name, value)


class LoggingSavingRecord(SavingRecord):
    def __setattr__(self, name, value):
        print(f'* Called __setattr__({name!r}, {value!r})')
        super().__setattr__(name, value)


# data = LoggingSavingRecord()
# print('Before:', data.__dict__)
# data.foo = 5
# print('After:', data.__dict__)
# data.foo = 7
# print('Final:', data.__dict__)


class BrokenDictionaryRecord:
    def __init__(self):
        self._data = {}

    def __getattribute__(self, name):
        print(f'* Called __getattribute__({name})')
        return self._data[name]


# data = BrokenDictionaryRecord()
# print(data.foo)


class DictionaryRecord:
    def __init__(self, data):
        # 这样写可以避免递归
        super().__setattr__("_data", data)

    def __getattribute__(self, name):
        print(f'* Called __getattribute__({name!r})')
        data_dict = super().__getattribute__('_data')
        return data_dict[name]

    def __setattr__(self, name, value):
        data_dict = super().__getattribute__('_data')
        data_dict[name] = value


data = DictionaryRecord({'foo': 3})
print('foo : ', data.foo)
data.bar = 5
print('bar:', data.bar)
data.bar = 8
print('updated bar:', data.bar)





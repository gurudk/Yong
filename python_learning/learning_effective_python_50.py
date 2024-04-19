"""

第50条 用__set_name__给类属性加注释

"""


# class Field:
#     def __init__(self):
#         self.name = None
#         self.internal_name = None
#
#     def __get__(self, instance, owner):
#         if instance is None:
#             return self
#
#         return getattr(instance, self.internal_name, '')
#
#     def __set__(self, instance, value):
#         setattr(instance, self.internal_name, value)


# class Customer:
#     first_name = Field('first_name')
#     last_name = Field('last_name')
#     prefix = Field('prefix')
#     suffix = Field('suffix')


# cust = Customer()
# print(f'Before:{cust.first_name!r}{cust.__dict__}')
# cust.first_name = 'Euclid'
# print(f'After:{cust.first_name!r}{cust.__dict__}')


# class Meta(type):
#     def __new__(meta, name ,bases, class_dict):
#         for key ,value in class_dict.items():
#             if isinstance(value, Field):
#                 value.name = key
#                 value.internal_name = '_' + key
#         cls = type.__new__(meta, name, bases, class_dict)
#         return cls
#
#
# class DatabaseRow(metaclass=Meta):
#     pass
#
#
# class BetterCustomer(DatabaseRow):
#     first_name = Field()
#     last_name = Field()
#     prefix = Field()
#     suffix = Field()
#
#
# cust = BetterCustomer()
# print(f'Before:{cust.first_name!r}{cust.__dict__}')
# cust.first_name = 'Euclid'
# print(f'After:{cust.first_name!r}{cust.__dict__}')


class Field:
    def __init__(self):
        self.name = None
        self.internal_name = None

    def __set_name__(self, owner, name):
        self.name = name
        self.internal_name = '_' + name

    def __get__(self, instance, owner):
        if instance is None:
            return self

        return getattr(instance, self.internal_name, '')

    def __set__(self, instance, value):
        setattr(instance, self.internal_name, value)


class FixedCustomer:
    first_name = Field()
    last_name = Field()
    prefix = Field()
    suffix = Field()


cust = FixedCustomer()
print(f'Before:{cust.first_name!r}{cust.__dict__}')
cust.first_name = 'Euclid'
print(f'After:{cust.first_name!r}{cust.__dict__}')
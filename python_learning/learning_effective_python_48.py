"""

第48条 用__init_subclass__验证子类是否写的准确

"""


class Meta(type):
    def __new__(meta, name, bases, class_dict):
        print(f'* Running {meta}.__new__ for {name}')
        print('Bases:', bases)
        print(class_dict)
        return type.__new__(meta, name, bases, class_dict)


class MyClass(metaclass=Meta):
    stuff = 123

    def foo(self):
        pass


class MySubclass(MyClass):
    other = 367

    def bar(self):
        pass


class ValidatePolygon(type):
    def __new__(meta, name, bases, class_dict):
        if bases:
            if class_dict['sides'] < 3:
                raise ValueError('Polygons need 3+ sides')
        return type.__new__(meta, name ,bases, class_dict)


class Polygon(metaclass=ValidatePolygon):
    sides = None

    @classmethod
    def interior_angles(cls):
        return (cls.sides - 2) * 180


class Triangle(Polygon):
    sides = 3


class Rectangle(Polygon):
    sides = 4


class Nonagon(Polygon):
    sides = 9


assert Triangle.interior_angles() == 180
assert Rectangle.interior_angles() == 360
assert Nonagon.interior_angles() == 1260


# print('Before class')
#
#
# class Line(Polygon):
#     print('Before sides')
#     sides = 2
#     print('After sides')
#
#
# print('After class')


class BetterPolygon:
    sides = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__()
        if cls.sides < 3:
            raise ValueError('Polygons need 3+ sides')

    @classmethod
    def interior_angles(cls):
        return (cls.sides - 2) * 180


class Hexgon(BetterPolygon):
    sides = 6


assert Hexgon.interior_angles() == 720

# print('Before class')
#
#
# class Point(BetterPolygon):
#     sides = 1
#
#
# print('After class')


class ValidateFilled(type):
    def __new__(meta, name, bases, class_dict):
        if bases:
            if class_dict['color'] not in ('red', 'green'):
                raise ValueError('Fill color must be supported')
        return type.__new__(meta, name, bases, class_dict)


# class Filled(metaclass=ValidateFilled):
#     color = None


# class RedPentagon(Filled, Polygon):
#     color = 'red'
#     sides = 5


class ValidatePolygon(type):
    def __new__(meta, name , bases, class_dict):
        if not class_dict.get('is_root'):
            if class_dict['sides'] < 3:
                raise ValueError('Polygons need 3+ sides')

        return type.__new__(meta, name, bases, class_dict)


class Polygon(metaclass=ValidatePolygon):
    is_root = True
    sides = None


class ValidateFilledPolygon(ValidatePolygon):
    def __new__(meta, name, bases, class_dict):
        if not class_dict.get('is_root'):
            if class_dict['color'] not in ['red', 'green']:
                raise ValueError('Fill color must be supported')
        return super().__new__(meta, name, bases, class_dict)


#  元类也可以使用继承树来扩展验证逻辑
class FilledPolygon(Polygon, metaclass=ValidateFilledPolygon):
    is_root = True
    color = None


class GreenPentagon(FilledPolygon):
    color = 'green'
    sides = 5


greenie = GreenPentagon()
assert isinstance(greenie, Polygon)

####################################################################################
# 使用__init_subclass__来执行验证逻辑，这样元类感觉有点尴尬阿
####################################################################################


class Filled:
    color = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__()
        if cls.color not in ('red', 'green', 'blue', 'beige'):
            raise ValueError('Fills need a valid color')


class RedTriangle(Filled, BetterPolygon):
    color = 'red'
    sides = 3


ruddy = RedTriangle()
assert isinstance(ruddy, Filled)
assert isinstance(ruddy, BetterPolygon)


class BlueLine(Filled, BetterPolygon):
    color = 'blue'
    sides = 3


class BeigeSquare(Filled, BetterPolygon):
    color = 'beige'
    sides = 4


class Top:
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__()
        print(f'Top for {cls}')


class Left(Top):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__()
        print(f'Left for {cls}')


class Right(Top):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__()
        print(f'Right for {cls}')


class Bottom(Left, Right):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__()
        print(f'Bottom for {cls}')
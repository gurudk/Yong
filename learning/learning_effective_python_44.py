"""

第44条 用春属性修饰器取代传统的getter和setter方法

"""


class Resistor:
    def __init__(self, ohms):
        self.ohms = ohms
        self.voltage = 0
        self.current = 0


r1 = Resistor(50e3)
r1.ohms = 10e3


class VoltageResistance(Resistor):
    def __init__(self,ohms):
        super().__init__(ohms)
        self._voltage = 0

    @property
    def voltage(self):
        return self._voltage

    @voltage.setter
    def voltage(self, voltage):
        self._voltage = voltage
        self.current = self._voltage / self.ohms


r2 = VoltageResistance(1e3)
print(f'Before :{r2.current:.2f} amps')
r2.voltage = 10
print(f'After :{r2.current:.2f} amps')


class BoundedResistance(Resistor):
    def __init__(self, ohms):
        super().__init__(ohms)

    @property
    def ohms(self):
        return self._ohms

    @ohms.setter
    def ohms(self, ohms):
        if ohms <= 0:
            raise ValueError(f'ohms must be > 0; got {ohms}')
        self._ohms = ohms


r3 = BoundedResistance(1e3)
r3.ohms = 0.1


class FixedResistance(Resistor):
    def __init__(self, ohms):
        super().__init__(ohms)

    @property
    def ohms(self):
        return self._ohms

    @ohms.setter
    def ohms(self, ohms):
        if hasattr(self, '_ohms'):
            raise AttributeError('Ohms is immutable')
        self._ohms = ohms


r4 = FixedResistance(1e3)
r4.ohms = 2e3






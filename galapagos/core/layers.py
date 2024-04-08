from galapagos.core.variable import Parameter


class Layer:
    def __init__(self):
        self._params = set()

    def __setattr__(self, key, value):
        if isinstance(value, Parameter):
            self._params.add(key)

        super().__setattr__(key, value)
   
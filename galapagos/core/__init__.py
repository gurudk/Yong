from galapagos.core.variable import Function
from galapagos.core.variable import Variable
from galapagos.core.variable import as_array
from galapagos.core.variable import as_variable
from galapagos.core.variable import no_grad
from galapagos.core.variable import setup_variable
from galapagos.core.variable import using_config
from galapagos.core.variable import Config

from galapagos.core.models import Model
from galapagos.core.layers import Layer

import galapagos.core.functions as F

Variable.__getitem__ = F.get_item

setup_variable()

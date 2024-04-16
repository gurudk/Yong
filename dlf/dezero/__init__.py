# =============================================================================
# step23.pyからstep32.pyまではsimple_coreを利用
is_simple_core = False  # True
# =============================================================================

if is_simple_core:
    from dlf.dezero.core_simple import Variable
    from dlf.dezero.core_simple import Function
    from dlf.dezero.core_simple import using_config
    from dlf.dezero.core_simple import no_grad
    from dlf.dezero.core_simple import as_array
    from dlf.dezero.core_simple import as_variable
    from dlf.dezero.core_simple import setup_variable

else:
    from dlf.dezero.core import Variable
    from dlf.dezero.core import Parameter
    from dlf.dezero.core import Function
    from dlf.dezero.core import using_config
    from dlf.dezero.core import no_grad
    from dlf.dezero.core import test_mode
    from dlf.dezero.core import as_array
    from dlf.dezero.core import as_variable
    from dlf.dezero.core import setup_variable
    from dlf.dezero.core import Config
    from dlf.dezero.layers import Layer
    from dlf.dezero.models import Model
    from dlf.dezero.datasets import Dataset
    from dlf.dezero.dataloaders import DataLoader
    from dlf.dezero.dataloaders import SeqDataLoader

    import dlf.dezero.datasets
    import dlf.dezero.dataloaders
    import dlf.dezero.optimizers
    import dlf.dezero.functions
    import dlf.dezero.functions_conv
    import dlf.dezero.layers
    import dlf.dezero.utils
    import dlf.dezero.cuda
    import dlf.dezero.transforms

setup_variable()
__version__ = '0.0.13'

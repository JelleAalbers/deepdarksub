import types
from pathlib import Path
import manada

from .utils import exporter, load_py_file
export, __all__ = exporter()

__all__.extend(['MANADA_ROOT', 'manada'])

MANADA_ROOT = Path(manada.__file__).parent.parent


@export
def load_manada_config(x=None):
    if isinstance(x, types.ModuleType):
        # This is already a loaded manada config
        return x

    config_name = 'config_d_los_sigma_sub' if x is None else x
    if config_name.endswith('.py'):
        config_name = x[:-3]
    return load_py_file(
        MANADA_ROOT / 'configs' / (config_name + '.py'),
        'manada_config_' + config_name)

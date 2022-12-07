import types
from pathlib import Path
import paltas

from .utils import exporter, load_py_file
export, __all__ = exporter()

__all__.extend(['PALTAS_ROOT', 'paltas'])

PALTAS_ROOT = Path(paltas.__file__).parent


@export
def load_paltas_config(x=None):
    if isinstance(x, types.ModuleType):
        # This is already a loaded paltas config
        return x
    if x is None:
        # Load a default config
        x = 'config_train'

    if x.endswith('.py'):
        # This is a path to a config
        # (maybe inside paltas, maybe not)
        config_path = Path(x)
        config_name = Path(x).stem
    else:
        # This is one of paltas's named configs
        config_name = x
        config_path = PALTAS_ROOT / 'Configs' / 'acs' / (x + '.py')
    assert config_path.exists()

    return load_py_file(
        str(config_path),
        'paltas_config_' + config_name)


@export
def take_config_medians(config):
    """Replaces distribution.rvs entries with their medians"""
    return {
        k: v if not callable(v) else v.__self__.median()
        for k, v in config.items()}

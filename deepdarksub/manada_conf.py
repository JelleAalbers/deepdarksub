import types
from pathlib import Path
import manada

from .utils import exporter, load_py_file
export, __all__ = exporter()

__all__.extend(['MANADA_ROOT', 'manada'])

MANADA_ROOT = Path(manada.__file__).parent


@export
def load_manada_config(x=None):
    if isinstance(x, types.ModuleType):
        # This is already a loaded manada config
        return x
    if x is None:
        # Load a default config
        x = 'config_amp_and_slope'

    if x.endswith('.py'):
        # This is a path to a config
        # (maybe inside manada, maybe not)
        config_path = Path(x)
        config_name = Path(x).stem
    else:
        # This is one of manada's named configs
        config_name = x
        config_path = MANADA_ROOT / 'Configs' / (x + '.py')
    assert config_path.exists()

    return load_py_file(
        str(config_path),
        'manada_config_' + config_name)


@export
def take_config_medians(config):
    """Replaces distribution.rvs entries with their medians"""
    return {
        k: v if not callable(v) else v.__self__.median()
        for k, v in config.items()}
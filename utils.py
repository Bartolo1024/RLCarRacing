import functools
from typing import Callable

import yaml


def unpack_config(func: Callable) -> Callable:
    """Load parameters from a config file and inject it to function keyword arguments"""
    @functools.wraps(func)
    def _wrapper(*args, **kwargs):
        config_file = kwargs.get('config')
        if config_file:
            del kwargs['config']
            with open(config_file, 'r') as f:
                run_args = yaml.full_load(f)
                kwargs.update(run_args)
        ret = func(*args, **kwargs)
        return ret

    return _wrapper